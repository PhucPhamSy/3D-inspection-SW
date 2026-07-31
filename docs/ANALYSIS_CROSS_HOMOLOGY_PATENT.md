# 3D Analysis — Hierarchical Homologous Cross-Comparison (H²C²)

**Status:** Concept + patent-oriented method design (not legal filing text)  
**Product surface:** 3D Analysis tab · *Metrology Lab*  
**Related:** [`ANALYSIS_SOH_REDESIGN.md`](ANALYSIS_SOH_REDESIGN.md), [`BATCH_REVIEW_CONCEPT_REDESIGN.md`](BATCH_REVIEW_CONCEPT_REDESIGN.md)  
**Mockup:** [`mockups/tabs_redesign_preview.html`](mockups/tabs_redesign_preview.html) → studio **Homology Cross**

> **Disclaimer:** This document describes an engineering invention concept for product/IP discussion.  
> It is **not** a formal patent application. Claim language should be refined by patent counsel.

---

## 0. What is fair to compare (and what is not)

### 0.1 Chips are design-identical — Chip A vs Chip B at same FOV is fully valid

On a production wafer, **every die is the same design**: same bump lattice, same FOV tile map (P1–P9), same layer stack.

So this is not only allowed — it is the **primary** cross-compare:

```text
WAFER01 / Chip_11_11 / FOV_P6   vs   WAFER01 / Chip_19_11 / FOV_P6
```

Same FOV index ⇒ same *volume class* (same tile on the die layout).  
Differences in SOH / void / gap are then **process / position-on-wafer / tool**, not “different product design.”

**Folder map is enough to join at FOV level:**

```text
.../Chip_{col}_{row}/FOV_P{n}/Results/...
         │                │
         │                └─ FOV index n  →  must match for die-to-die
         └─ chip instance (free in M1)
```

Bump-level join inside that FOV still uses lattice index `(grid_row, grid_col)` + `layer` (same design sites in every chip).

### 0.2 What must NOT be mixed without a special mode

| Compare | Valid? | Why |
|---------|--------|-----|
| Chip A **P6** vs Chip B **P6** | ✅ Yes | Same design tile; dies are copies |
| Chip A **P6** vs Chip A **P4** | ⚠ Special | Same die, **different** FOV tile / volume class |
| Chip A whole-die mean vs Chip B whole-die mean (mix all P) | ❌ Weak | Mixes FOV classes unless you pool after per-FOV normalize |
| Wafer01 Chip_19_11 P6 vs Wafer09 Chip_19_11 P6 | ✅ Yes | Same chip address + same FOV on design map |

| Factor | Why FOV P4 ≠ FOV P6 (even on same die) |
|--------|----------------------------------------|
| Scan geometry | Beam path, tilt, reconstruction ROI differ by field |
| Volume SNR / artifact | Center FOV vs edge FOV → different noise, cupping, partial volume |
| Local process | Die edge vs center: plating, warpage, stack stress |
| Lattice context | Neighbor density, pad pattern may differ by FOV tile |

**Clarification:** we never said “chips cannot be compared.”  
We said: **do not compare across different FOV indices** as if they were the same volume — unless the goal is FOV-bias study (M5).

**Core insight (product + patent seed):**

> Dies are identical by design → folder path `Chip_*/FOV_Pn` is a natural homology map.  
> Join FOV by `n`, join bumps by lattice + layer, free the chip (or wafer / FOUP / lot) to measure process.

---

## 1. Site Homology Key (SHK) — the atomic identity

### 1.1 Definition

Every measured bump (MES object) is assigned:

```text
SHK = (layer_id, fov_index, grid_row, grid_col)
```

Optional extension when recipe has multi-pattern:

```text
SHK+ = (recipe_id, layer_id, fov_index, grid_row, grid_col [, sub_site])
```

| Component | Meaning | Example |
|-----------|---------|---------|
| `layer_id` | Stack layer L1…L12 | `L8` |
| `fov_index` | FOV point on die P1–P9 (or recipe map) | `P6` |
| `grid_row, grid_col` | Bump lattice index **inside that FOV** | `(3, 2)` |

**Not part of SHK** (these are *stratum* / instance axes):

```text
date, lot_id, foup_id, wafer_id, chip_col, chip_row, run_id, timestamp
```

### 1.2 Why grid index, not XYZ centroid

- Centroids drift with alignment residual → false mismatch across chips.  
- Lattice index is **recipe-defined homology**: “the same designed bump site in the FOV tile.”  
- Centroid used only for **within-FOV spatial stats** (Moran, SII), not for cross-chip join.

### 1.3 Homology join

Two MES rows are a **matched pair** iff `SHK(a) == SHK(b)` and stratum keys differ as allowed by the compare mode.

```text
MATCHED_PAIR(a,b | mode) ⇔
    SHK(a) = SHK(b)
    ∧ stratum_diff(a,b) allowed by mode
    ∧ recipe_compatible(a,b)
```

### 1.4 Completeness / coverage

For a comparison set \(S\) of FOV runs:

\[
\mathrm{Coverage}(\mathrm{SHK}) = \frac{\#\{\text{runs in } S \text{ that contain SHK}\}}{|S_{\mathrm{eligible}}|}
\]

UI shows **matched N**, **missing rate**, **unmatched bumps** (orphan lattice sites).  
Never silently average over incomplete keys.

---

## 2. Hierarchy strata (where cross lives)

```text
LOT
 └─ FOUP
     └─ WAFER
         └─ CHIP (col,row)
             └─ FOV run (P1..P9)     ← volume instance
                 └─ LAYER
                     └─ BUMP site (r,c)   ← SHK leaf
```

### 2.1 Compare modes (operator intent → mathematical lock)

| Mode ID | Lock (must equal) | Free (may differ) | Question answered |
|---------|-------------------|-------------------|-------------------|
| **M1 Die-to-die** ⭐ default | SHK + wafer + FOUP + lot | **chip** | Same wafer: Chip A vs B at **same FOV_Pn** (dies identical) |
| **M2 Wafer-to-wafer** | SHK + chip_coord* + FOUP + lot | wafer | Same map address + FOV, other wafers |
| **M3 FOUP-to-FOUP** | SHK + chip_coord* + wafer_slot* + lot | FOUP | Carrier / FOUP effect? |
| **M4 Lot-to-lot** | SHK + chip_coord* + wafer_slot* + FOUP_slot* | lot | Process / lot shift? |
| **M5 FOV-position effect** | layer + grid + chip + wafer… | **fov_index only** | *Within one die*: does P-position bias SOH? |
| **M6 Full nested** | SHK | all strata | Variance components at every level |

\* *chip_coord* / *wafer_slot* / *FOUP_slot* = same **relative** position in the map (e.g. Chip 19,11 on every wafer), not absolute path string.

### 2.2 Folder → join map (production tree)

```text
{date}/
  {LOT}_{FOUP}/                 e.g. LOT240101_F09
    {WAFER}/                    e.g. WAFER01
      Chip_{col}_{row}/
        FOV_P{n}/               ← join key for “same FOV”
          Input.tif
          Results/
            Layer_{k}/ …        ← layer in SHK
            (MES rows: grid r,c)
```

| Level | Folder / field | M1 Die↔Die |
|-------|----------------|------------|
| Lot+FOUP | `LOT240101_F09` | same |
| Wafer | `WAFER01` | same |
| Chip | `Chip_11_11` vs `Chip_19_11` | **free** (the compare) |
| FOV | `FOV_P6` = `FOV_P6` | **locked** |
| Layer + grid | MES / Results | **locked** (same designed bump) |

**Default for PE:** **M1** — pick any chips on the wafer, lock `FOV_Pn`, compare bump-by-bump (or FOV aggregate after per-site match).  
Dies are copies; folder names are the practical homology index.

---

## 3. Method: Hierarchical Homologous Cross-Comparison (H²C²)

### 3.1 Data tensor

Build a sparse tensor:

\[
X[\underbrace{\ell, f, i, j}_{\mathrm{SHK}},\; \underbrace{L, U, W, C, t}_{\mathrm{stratum}}] = m
\]

- \(\ell\) layer, \(f\) FOV index, \((i,j)\) grid  
- \(L\) lot, \(U\) FOUP, \(W\) wafer, \(C\) chip, \(t\) time/run  
- \(m\) metric: SOH, void%, bump vol, gap, judgment∈{0,1}, …

### 3.2 Homologous delta

For two strata \(A, B\) under mode \(M\):

\[
\Delta_m(\mathrm{SHK}) = m_A(\mathrm{SHK}) - m_B(\mathrm{SHK})
\]

only where both exist. Aggregate:

| Aggregate | Use |
|-----------|-----|
| mean Δ | systematic shift |
| median Δ | robust shift |
| σ(Δ) | site-wise instability of the *difference* |
| P95 \|Δ\| | worst-site engineering limit |
| % sites \|Δ\| > tol | homology fail rate |

### 3.3 Nested variance decomposition (patent-friendly core)

For fixed SHK (or pooled over SHK with FOV locked), nested model:

\[
m = \mu_{\mathrm{SHK}}
  + \alpha_{\mathrm{Lot}}
  + \beta_{\mathrm{FOUP|Lot}}
  + \gamma_{\mathrm{Wafer|FOUP}}
  + \delta_{\mathrm{Chip|Wafer}}
  + \varepsilon_{\mathrm{run}}
\]

Estimate variance components:

\[
\sigma^2_{\mathrm{Lot}},\;
\sigma^2_{\mathrm{FOUP}},\;
\sigma^2_{\mathrm{Wafer}},\;
\sigma^2_{\mathrm{Chip}},\;
\sigma^2_{\mathrm{residual}}
\]

**Product readout (unique UI):**

```text
VARIANCE STACK @ SHK or @ FOV=P6, Layer=L8
  Lot      ████░░░░░░  18%
  FOUP     ██░░░░░░░░   7%
  Wafer    ██████░░░░  29%
  Chip     ████████░░  34%
  Residual ███░░░░░░░  12%
```

Interpretation engineers care about:

- High **Chip%** → die systematic (local process)  
- High **Wafer%** → wafer process / chuck / map  
- High **FOUP%** → logistics / FOUP handling  
- High **Lot%** → lot process shift  
- High **Residual** → metrology noise / re-run instability  

### 3.4 FOV-conditioned baseline (why volumes differ)

Estimate FOV fixed effect inside a stratum:

\[
m = \mu_{\mathrm{layer,grid}} + \phi_f + \text{higher terms}
\]

\(\phi_f\) = **FOV bias field** (P1…P9).  
Display as 3×3 bias map — *not* pass/fail.

**Claim seed:** separating **FOV-position metrology bias** from **true process difference at homologous sites**.

### 3.5 Homology Stability Index (HSI)

Per SHK across a cohort:

\[
\mathrm{HSI} = 1 - \frac{\sigma_{\mathrm{across\,strata}}(m)}{\sigma_{\mathrm{ref}} + \epsilon}
\]

or robust:

\[
\mathrm{HSI} = 1 - \frac{\mathrm{IQR}(m_{\mathrm{strata}})}{\mathrm{Tol}}
\]

Rank sites: **stable gold sites** vs **chronic hotspots**.  
Hotspots feed Root-cause / LOO.

### 3.6 Leave-One-Stratum-Out impact (LOSO-VD)

Generalize LOO-VD from “leave one sample” to **leave one chip / wafer / FOUP**:

\[
\mathrm{Impact}(s) = \mathcal{L}(S) - \mathcal{L}(S \setminus s)
\]

where \(\mathcal{L}\) = e.g. σ of homologous means, or %Tol fail rate.

Answers: “Which wafer is poisoning cross-lot σ?” without opening every FOV.

### 3.7 Pairwise homology matrix

For N chips (or N wafers) under locked SHK set:

\[
D_{ab} = \mathrm{median}_{\mathrm{SHK}} |m_a - m_b|
\quad\text{or}\quad
1 - \mathrm{corr}_{\mathrm{SHK}}(m_a, m_b)
\]

Heatmap **Chip×Chip** or **Wafer×Wafer** — clustering process twins vs outliers.

### 3.8 Recipe / lattice alignment gate

Before join:

1. Same `recipe_id` (or compatible lattice signature)  
2. Same FOV map definition (P1–P9 geometry)  
3. Grid max bounds compatible  
4. Optional: lattice checksum from teaching  

Fail gate → block cross-compare with explicit reason (patent: *safety interlock for invalid homology*).

---

## 4. Metrics catalog (what to put on SHK)

| Metric \(m\) | Source | Cross meaning |
|--------------|--------|----------------|
| SOH / Z height | MES | Primary stack health |
| Void% | MES | Local defect intensity |
| Bump volume | MES | Plating / geometry |
| B2B gap X/Y/Euc | boundary_gap.csv | Pitch stability **at lattice edge pairs** |
| Judgment | rule | Binary NG rate at site |
| Enhance SNR proxy | optional | Metrology confidence weight |

**Weighted homology** (advanced): weight Δ by inverse run residual variance or SNR so noisy FOVs don’t dominate lot-to-lot claims.

---

## 5. UX: “Homology Cross” studio (3D Analysis)

### 5.1 Placement

Replace vague **Compare** with two submodes:

| Studio tab | Role |
|------------|------|
| **Homology Cross** ⭐ | Matched SHK cross chip/wafer/FOUP/lot |
| **FOV Bias** | Deliberate unlock FOV (M5) — separate warning theme |
| **Variance Stack** | Nested σ% (may be panel inside Homology) |
| **Bump σ** | Per-SHK σ across set (classic) |
| **Root cause** | LOSO + HSI ranking |
| **Overview / Maps / Table** | Unchanged roles |

### 5.2 Homology Cross control bar

```text
Mode: [ Die↔Die ▼ ]   Anchor: [ Wafer01 · Chip 19,11 · as baseline ▼ ]
Lock: ☑ Layer  ☑ FOV index  ☑ Grid (r,c)     ← SHK locks (default all on)
Free: chip / wafer / FOUP / lot               ← driven by mode
Metric: [ SOH ▼ ]   Aggregate: [ median Δ ▼ ]
Spec Limits: LCL [ -1.5 µm ]  UCL [ +1.5 µm ]   K-factor: [ 5.15 ▼ ]
Grade Thresholds: Grade1 Limit [ 10% ]  Grade2 Limit [ 30% ]
  ➜ %Tol = (5.15 * stdDev) / |UCL - LCL| * 100%
  ➜ Chart Color: Green (< Grade1 Limit), Amber (Grade1..Grade2 Limit), Red (≥ Grade2 Limit)
```

### 5.3 Center views

1. **Δ heatmap by FOV (3×3)** — each cell = median |Δ| or mean Δ at that FOV (grids pooled or per selected layer)  
2. **Δ bump grid** — for selected FOV+layer, color by Δ vs anchor  
3. **Pair matrix** — Chip×Chip or Wafer×Wafer distance  
4. **Matched table** — SHK | m_anchor | m_peer | Δ | z-score | coverage  

### 5.4 Right insight

- Coverage % matched  
- Systematic shift (mean Δ)  
- Hotspot top-K SHK  
- Variance stack mini  
- Deep link: open both FOVs in Viewer (side-by-side future) / Line Pulse  

### 5.5 Wizard: “Build fair cohort”

```text
1) Pick Anchor FOV run
2) Auto-propose peers with same SHK support:
     · same wafer other chips (M1)
     · other wafers same chip coord (M2)
     · …
3) Show coverage matrix before Run
4) Run H²C²
```

This prevents PE from multi-selecting incompatible FOVs by accident.

---

## 6. Worked examples (from your mental model)

### Example A — same wafer, different chips, same FOV & bump ⭐ canonical

```text
Mode M1  (chips are design-identical)
Folder map:
  .../WAFER01/Chip_11_11/FOV_P6/   ↔   .../WAFER01/Chip_19_11/FOV_P6/
SHK = (L8, P6, r=3, c=2)
ΔSOH = SOH(B) − SOH(A)
```

**Valid and primary:** every chip is the same product die; same `FOV_P6` folder = same tile on that die; lattice (r,c) = same designed bump.

### Example B — different wafers, same chip coordinate & FOV

```text
Mode M2
SHK = (L8, P6, 3, 2)
A = WAFER01 / Chip_19_11 / P6
B = WAFER09 / Chip_19_11 / P6
```

**Valid:** wafer drift at homologous site (same die address on map).

### Example C — FOUP / Lot climb

```text
M3/M4: lock SHK + chip_coord + wafer_slot (e.g. slot order)
free FOUP or Lot
```

### Example D — invalid (UI should warn)

```text
Chip_11_11 / P4  vs  Chip_19_11 / P6   without FOV unlock
→ BLOCKED: FOV index not homologous (volume class mismatch)
```

Unless user enters **FOV Bias** mode M5 with banner:

```text
⚠ FOV bias study — volumes are NOT class-matched. Do not use for die yield blame.
```

---

## 7. Differentiation vs current Cross-σ

| Today (folder samples) | H²C² |
|------------------------|------|
| Sample = folder name | Sample = stratum path + SHK |
| Cross-σ across whole FOV means | Matched site Δ + nested σ |
| Easy to mix P4 with P6 | FOV locked by default |
| No FOUP/Lot language | Explicit M1–M4 ladder |
| No coverage | Matched N + missing rate |
| LOO per sample folder | LOSO per chip/wafer/FOUP |
| — | FOV bias field φ_f separated |

---

## 8. Patent-oriented claim seeds (for counsel)

*Illustrative only — not filed claims.*

1. **A method** of comparing multi-die 3D semiconductor inspection measurements by:  
   (a) extracting object metrics from FOV volumes;  
   (b) assigning each object a **site homology key** comprising layer, FOV index, and lattice indices;  
   (c) joining objects across dies/wafers/FOUPs/lots **only when keys match**;  
   (d) computing homologous differences and/or nested variance components along a manufacturing hierarchy.

2. **A system** that **blocks or warns** on cross-comparison when FOV indices differ, unless a dedicated FOV-bias mode is selected.

3. **A method** of decomposing metric variance at fixed homology keys into lot, FOUP, wafer, chip, and residual components for HBM bump metrology.

4. **A method** of ranking lattice sites by a **homology stability index** across strata for process hotspot discovery.

5. **A UI/system** presenting a FOV-conditioned bias map separately from homologous cross-stratum delta maps.

6. **Weighted homologous aggregation** using per-FOV quality/SNR weights.

**Prior art to study (counsel):** nested ANOVA in semiconductor SPC, die-to-die matching, spatial homology in bioimaging, SEMI metrology standards — differentiation is **FOV-as-volume-class + bump lattice SHK + multi-stratum HBM hierarchy + enforced join gate**.

---

## 9. Implementation sketch (product, later)

### 9.1 DB

Ensure `mes_objects` always has: `layer_name, grid_row, grid_col, soh, ratio, …`  
Ensure `fov_runs` has: `lot_id, foup_id, wafer_id, chip_col, chip_row, fov_index, recipe_id`  
*(Parse `LOT240701_F01` → lot + foup at write time.)*

Index:

```sql
CREATE INDEX IF NOT EXISTS idx_mes_shk
  ON mes_objects(layer_name, /* join run for fov */ );
-- practical: materialised view homologous_site_metrics
```

### 9.2 Core API (Python)

```python
@dataclass(frozen=True)
class SiteHomologyKey:
    layer_id: str
    fov_index: int
    grid_row: int
    grid_col: int

@dataclass
class Stratum:
    lot_id: str
    foup_id: str
    wafer_id: str
    chip_col: int
    chip_row: int
    run_id: str

def build_homology_table(rows, mode) -> pd.DataFrame: ...
def homologous_delta(anchor, peers, metric) -> HomologyResult: ...
def nested_variance(table, shk_filter) -> VarianceStack: ...
def fov_bias_field(table, locks) -> FovBiasMap: ...
def hsi_rank(table) -> list[HSIRow]: ...
```

### 9.3 Phases

| Phase | Deliverable |
|-------|-------------|
| **P0** | SHK in data model; M1 Die↔Die Δ table + coverage |
| **P1** | M2 Wafer↔Wafer; Δ bump grid + FOV 3×3 |
| **P2** | Variance stack + HSI ranking |
| **P3** | M3/M4 FOUP/Lot; pair matrix; LOSO |
| **P4** | FOV Bias mode M5; SNR weights; export patent evidence pack |

---

## 10. One-sentence product doctrine

> **3D Analysis never compares volumes; it compares homologous sites across volumes, then attributes difference to the manufacturing hierarchy.**

Line Pulse still answers *pass/fail now*.  
Homology Cross answers *where the process moved, at the only sites that are fair to compare*.

---

## 11. Open technical decisions

1. Grid origin convention after teaching align — document in recipe.  
2. Missing bump at SHK: impute vs exclude (default **exclude** for Δ).  
3. Multi-run same FOV (retest): use latest / mean / gage pair mode.  
4. Layer name normalization (`Layer_8` vs `L8`).  
5. Whether B2B edges use SHK-pair keys `(src_r,src_c,dst_r,dst_c,fov,layer)`.

---

*End — Hierarchical Homologous Cross-Comparison (H²C²) concept for Inno3D Analysis / IP discussion.*
