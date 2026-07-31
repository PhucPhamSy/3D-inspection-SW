"""Hierarchical Homologous Cross-Comparison (H²C²) — domain types + engine.

P0/P1: Die↔Die homologous Δ (M1) with FOV lock gate and coverage.
P2: nested variance stack + HSI ranking (approximate variance-of-means).
P3: pairwise homology matrix + leave-one-stratum-out (LOSO) impact.
M2: Wafer↔Wafer lock (chip coord + lot_foup; free wafer) in homologous_delta.
See docs/ANALYSIS_CROSS_HOMOLOGY_PATENT.md.
"""
from __future__ import annotations

import math
import re
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from inno3d.core.lot_foup import join_lot_foup, split_lot_foup


@dataclass(frozen=True)
class SiteHomologyKey:
    """Atomic identity for a designed bump site (SHK)."""

    layer_id: str
    fov_index: int
    grid_row: int
    grid_col: int


@dataclass(frozen=True)
class Stratum:
    """Manufacturing hierarchy instance axes (not part of SHK)."""

    lot_id: str
    foup_id: str
    wafer_id: str
    chip_col: int
    chip_row: int
    run_id: str


class CompareMode(str, Enum):
    """Operator compare intent → lock/free strata (patent modes M1–M6)."""

    M1_DIE = "M1"  # lock SHK+wafer+FOUP+lot; free chip
    M2_WAFER = "M2"
    M3_FOUP = "M3"
    M4_LOT = "M4"
    M5_FOV_BIAS = "M5"  # unlock fov_index; warn banner required
    M6_NESTED = "M6"


@dataclass
class HomologyResult:
    mode: CompareMode
    metric: str
    matched_n: int
    missing_rate: float
    rows: List[Dict[str, Any]] = field(default_factory=list)
    orphans_anchor: int = 0
    orphans_peer: int = 0
    mean_delta: float = 0.0
    median_delta: float = 0.0
    p95_abs_delta: float = 0.0
    blocked: bool = False
    block_reason: str = ""


def normalize_layer_id(name: str) -> str:
    """Normalize layer labels to ``L{n}`` when a number is present.

    Examples: ``Layer_8`` / ``L8`` / ``layer 8`` → ``L8``.
    Unknown shapes are returned stripped upper-cased lightly.
    """
    s = (name or "").strip()
    if not s:
        return ""
    m = re.search(r"(\d+)", s)
    if m:
        return f"L{int(m.group(1))}"
    return s


def shk_from_mes_row(row: Dict[str, Any]) -> Optional[SiteHomologyKey]:
    """Build SHK from a ``list_mes_joined`` dict. None if grid missing / invalid."""
    if not row:
        return None
    try:
        gr = int(row.get("grid_row")) if row.get("grid_row") is not None else -1
        gc = int(row.get("grid_col")) if row.get("grid_col") is not None else -1
    except (TypeError, ValueError):
        return None
    if gr < 0 or gc < 0:
        return None
    try:
        fi = int(row.get("fov_index") or 0)
    except (TypeError, ValueError):
        fi = 0
    if fi <= 0:
        return None
    layer = normalize_layer_id(str(row.get("layer_name") or ""))
    if not layer:
        return None
    return SiteHomologyKey(
        layer_id=layer,
        fov_index=fi,
        grid_row=gr,
        grid_col=gc,
    )


def stratum_from_mes_row(row: Dict[str, Any]) -> Stratum:
    """Build Stratum from joined MES/run meta. Uses lot_foup split when needed."""
    lot_id = str(row.get("lot_id") or "").strip()
    foup_id = str(row.get("foup_id") or "").strip()
    if not lot_id or not foup_id:
        lf = str(row.get("lot_foup_id") or "")
        lot_p, foup_p = split_lot_foup(lf)
        lot_id = lot_id or lot_p
        foup_id = foup_id or foup_p
    try:
        cc = int(row.get("chip_col") or 0)
    except (TypeError, ValueError):
        cc = 0
    try:
        cr = int(row.get("chip_row") or 0)
    except (TypeError, ValueError):
        cr = 0
    return Stratum(
        lot_id=lot_id,
        foup_id=foup_id,
        wafer_id=str(row.get("wafer_id") or ""),
        chip_col=cc,
        chip_row=cr,
        run_id=str(row.get("run_id") or ""),
    )


# ---------------------------------------------------------------------------
# M1 engine (C-P0)
# ---------------------------------------------------------------------------


def _metric_value(row: Dict[str, Any], metric: str) -> Optional[float]:
    """Parse a numeric metric from a MES / homology-table row. None if missing."""
    key = (metric or "soh").strip().lower()
    raw = row.get(key)
    if raw is None and key == "soh":
        raw = row.get("SOH") or row.get("SOH (um)")
    if raw is None and key == "ratio":
        raw = row.get("Ratio")
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _layers_and_fovs(rows: Iterable[Dict[str, Any]]) -> Tuple[Set[str], Set[int]]:
    layers: Set[str] = set()
    fovs: Set[int] = set()
    for row in rows or ():
        shk = shk_from_mes_row(row)
        if shk is None:
            # Still collect raw FOV / layer hints when SHK incomplete
            layer = normalize_layer_id(str(row.get("layer_name") or ""))
            if layer:
                layers.add(layer)
            try:
                fi = int(row.get("fov_index") or 0)
            except (TypeError, ValueError):
                fi = 0
            if fi > 0:
                fovs.add(fi)
            continue
        layers.add(shk.layer_id)
        fovs.add(shk.fov_index)
    return layers, fovs


def recipe_compatible(
    rows_a: Sequence[Dict[str, Any]],
    rows_b: Sequence[Dict[str, Any]],
) -> Tuple[bool, str]:
    """Gate: FOV map support / layer overlap. Fail → ``(False, reason)``.

    P0 checks (no recipe_id in schema):
      1. Both sides expose at least one FOV index (map present) and one layer.
      2. Normalized layer-name sets intersect.

    FOV *index equality* is enforced later by ``homologous_delta`` (M1–M4 lock);
    this gate only verifies both sides have a usable FOV map + shared layers.
    """
    layers_a, fovs_a = _layers_and_fovs(rows_a)
    layers_b, fovs_b = _layers_and_fovs(rows_b)

    if not layers_a or not fovs_a:
        return False, "Side A has no usable layer/FOV map (empty or invalid MES rows)"
    if not layers_b or not fovs_b:
        return False, "Side B has no usable layer/FOV map (empty or invalid MES rows)"

    layer_overlap = layers_a & layers_b
    if not layer_overlap:
        return (
            False,
            f"No layer overlap after normalize "
            f"(A={sorted(layers_a)}, B={sorted(layers_b)})",
        )

    return True, ""

def build_homology_table(
    rows: List[Dict[str, Any]],
    mode: CompareMode,
) -> List[Dict[str, Any]]:
    """Flatten joined MES rows into homology-ready records.

    Returns ``list[dict]`` (pandas is not a hard dependency in ``inno3d``).
    Columns: SHK (layer_id, fov_index, grid_row, grid_col), stratum
    (lot_id, foup_id, wafer_id, chip_col, chip_row, run_id, lot_foup_id),
    metrics (soh, ratio), plus ``mode``.

    Rows where ``shk_from_mes_row`` returns None are skipped (never imputed).
    Lot/FOUP parsed via ``split_lot_foup(lot_foup_id)`` when split fields absent.
    """
    out: List[Dict[str, Any]] = []
    mode_val = mode.value if isinstance(mode, CompareMode) else str(mode)
    for row in rows or ():
        shk = shk_from_mes_row(row)
        if shk is None:
            continue
        st = stratum_from_mes_row(row)
        lot_foup = join_lot_foup(st.lot_id, st.foup_id) or str(
            row.get("lot_foup_id") or ""
        )
        soh = _metric_value(row, "soh")
        ratio = _metric_value(row, "ratio")
        out.append(
            {
                "layer_id": shk.layer_id,
                "fov_index": shk.fov_index,
                "grid_row": shk.grid_row,
                "grid_col": shk.grid_col,
                "lot_id": st.lot_id,
                "foup_id": st.foup_id,
                "lot_foup_id": lot_foup,
                "wafer_id": st.wafer_id,
                "chip_col": st.chip_col,
                "chip_row": st.chip_row,
                "run_id": st.run_id,
                "soh": soh,
                "ratio": ratio,
                "mode": mode_val,
            }
        )
    return out


def _run_fov_index(run_rows: Sequence[Dict[str, Any]]) -> Optional[int]:
    for row in run_rows:
        try:
            fi = int(row.get("fov_index") or 0)
        except (TypeError, ValueError):
            fi = 0
        if fi > 0:
            return fi
    return None


def _run_stratum_keys(run_rows: Sequence[Dict[str, Any]]) -> Tuple[str, str]:
    """Return (wafer_id, lot_foup_id) from the first usable row."""
    for row in run_rows:
        st = stratum_from_mes_row(row)
        lf = join_lot_foup(st.lot_id, st.foup_id) or str(row.get("lot_foup_id") or "")
        return st.wafer_id, lf
    return "", ""


def _run_chip_coord(run_rows: Sequence[Dict[str, Any]]) -> Tuple[int, int]:
    """Return (chip_col, chip_row) from the first usable row."""
    for row in run_rows:
        st = stratum_from_mes_row(row)
        return st.chip_col, st.chip_row
    return 0, 0


def _shk_metric_map(
    run_rows: Sequence[Dict[str, Any]],
    metric: str,
) -> Dict[SiteHomologyKey, float]:
    """Map SHK → metric; missing SHK excluded; duplicate SHK keeps first value."""
    out: Dict[SiteHomologyKey, float] = {}
    for row in run_rows:
        shk = shk_from_mes_row(row)
        if shk is None:
            continue
        if shk in out:
            continue
        val = _metric_value(row, metric)
        if val is None:
            continue
        out[shk] = val
    return out


def _percentile(sorted_vals: Sequence[float], pct: float) -> float:
    """Linear-interpolation percentile on a pre-sorted ascending sequence."""
    n = len(sorted_vals)
    if n == 0:
        return 0.0
    if n == 1:
        return float(sorted_vals[0])
    rank = (pct / 100.0) * (n - 1)
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return float(sorted_vals[lo])
    w = rank - lo
    return float(sorted_vals[lo]) * (1.0 - w) + float(sorted_vals[hi]) * w


def _pair_delta_rows(
    anchor_map: Dict[SiteHomologyKey, float],
    peer_map: Dict[SiteHomologyKey, float],
    peer_run_id: str,
) -> Tuple[List[Dict[str, Any]], int, int]:
    """Match SHKs; return (delta rows without z_score, orphans_a, orphans_p)."""
    keys_a = set(anchor_map)
    keys_b = set(peer_map)
    matched = sorted(keys_a & keys_b, key=lambda k: (k.layer_id, k.fov_index, k.grid_row, k.grid_col))
    orphans_a = len(keys_a - keys_b)
    orphans_b = len(keys_b - keys_a)
    rows: List[Dict[str, Any]] = []
    for shk in matched:
        m_a = anchor_map[shk]
        m_b = peer_map[shk]
        delta = m_a - m_b
        rows.append(
            {
                "layer_id": shk.layer_id,
                "fov_index": shk.fov_index,
                "grid_row": shk.grid_row,
                "grid_col": shk.grid_col,
                "peer_run_id": peer_run_id,
                "m_anchor": m_a,
                "m_peer": m_b,
                "delta": delta,
                "z_score": 0.0,
                "coverage": 1.0,
            }
        )
    return rows, orphans_a, orphans_b


def _attach_z_scores(rows: List[Dict[str, Any]]) -> None:
    """In-place z-score of ``delta``; 0 when n < 2 or zero variance."""
    n = len(rows)
    if n < 2:
        for r in rows:
            r["z_score"] = 0.0
        return
    deltas = [float(r["delta"]) for r in rows]
    mean = statistics.mean(deltas)
    try:
        std = statistics.stdev(deltas)
    except statistics.StatisticsError:
        std = 0.0
    if std == 0.0 or not math.isfinite(std):
        for r in rows:
            r["z_score"] = 0.0
        return
    for r in rows:
        r["z_score"] = (float(r["delta"]) - mean) / std


def homologous_delta(
    rows: List[Dict[str, Any]],
    *,
    mode: CompareMode,
    anchor_run_id: str,
    peer_run_ids: List[str],
    metric: str = "soh",
    aggregate: str = "median",  # mean|median — used for summary stats only
) -> HomologyResult:
    """Compute homologous Δ between an anchor FOV run and peer run(s).

    M1 rules
    --------
    lock: layer, fov_index, grid_r/c, wafer_id, lot_foup (same wafer)
    free: chip_col / chip_row

    M2 rules (Wafer↔Wafer)
    ---------------------
    lock: layer, fov_index, grid_r/c, chip_col/row, lot_foup
    free: wafer_id

    Missing SHK: EXCLUDE (never impute).

    FOV gate: block if any peer ``fov_index`` ≠ anchor unless ``mode == M5``.

    Multi-peer (P0)
    ---------------
    Prefer exactly one ``peer_run_id`` for a clean single Δ table.
    If multiple peers are supplied, pairwise Δ rows are emitted for **each**
    peer (tagged with ``peer_run_id``); summary stats (mean/median/p95,
    orphans, missing_rate) are filled vs the **first** peer only.
    ``aggregate`` selects which central tendency is emphasized for callers
    that read a single summary; both ``mean_delta`` and ``median_delta`` are
    always populated.
    """
    if (aggregate or "median").strip().lower() not in ("mean", "median"):
        empty = HomologyResult(
            mode=mode if isinstance(mode, CompareMode) else CompareMode(str(mode)),
            metric=(metric or "soh").strip().lower(),
            matched_n=0,
            missing_rate=0.0,
            blocked=True,
            block_reason=f"aggregate must be 'mean' or 'median', got {aggregate!r}",
        )
        return empty
    mode = mode if isinstance(mode, CompareMode) else CompareMode(str(mode))
    metric_key = (metric or "soh").strip().lower()
    anchor_id = str(anchor_run_id or "").strip()
    peers = [str(p).strip() for p in (peer_run_ids or []) if str(p).strip()]

    empty = HomologyResult(mode=mode, metric=metric_key, matched_n=0, missing_rate=0.0)

    if not anchor_id:
        empty.blocked = True
        empty.block_reason = "anchor_run_id is required"
        return empty
    if not peers:
        empty.blocked = True
        empty.block_reason = "At least one peer_run_id is required"
        return empty
    if anchor_id in peers:
        empty.blocked = True
        empty.block_reason = "peer_run_ids must not include anchor_run_id"
        return empty

    by_run: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows or ():
        rid = str(row.get("run_id") or "").strip()
        if rid:
            by_run[rid].append(row)

    anchor_rows = by_run.get(anchor_id) or []
    if not anchor_rows:
        empty.blocked = True
        empty.block_reason = f"No rows for anchor_run_id={anchor_id!r}"
        return empty

    for pid in peers:
        if not by_run.get(pid):
            empty.blocked = True
            empty.block_reason = f"No rows for peer_run_id={pid!r}"
            return empty

    anchor_fov = _run_fov_index(anchor_rows)
    if anchor_fov is None:
        empty.blocked = True
        empty.block_reason = f"Anchor run {anchor_id!r} has no valid fov_index"
        return empty

    # FOV lock (unless M5 FOV-bias study)
    if mode != CompareMode.M5_FOV_BIAS:
        for pid in peers:
            peer_fov = _run_fov_index(by_run[pid])
            if peer_fov is None:
                empty.blocked = True
                empty.block_reason = f"Peer run {pid!r} has no valid fov_index"
                return empty
            if peer_fov != anchor_fov:
                empty.blocked = True
                empty.block_reason = (
                    f"FOV index not homologous under {mode.value}: "
                    f"anchor FOV_P{anchor_fov} vs peer {pid!r} FOV_P{peer_fov} "
                    f"(use mode M5 for FOV-bias study)"
                )
                return empty

    # M1 stratum lock: wafer + lot_foup (chip free)
    if mode == CompareMode.M1_DIE:
        a_wafer, a_lf = _run_stratum_keys(anchor_rows)
        for pid in peers:
            p_wafer, p_lf = _run_stratum_keys(by_run[pid])
            if a_wafer and p_wafer and a_wafer != p_wafer:
                empty.blocked = True
                empty.block_reason = (
                    f"M1 requires same wafer_id: anchor={a_wafer!r} vs peer={p_wafer!r}"
                )
                return empty
            if a_lf and p_lf and a_lf != p_lf:
                empty.blocked = True
                empty.block_reason = (
                    f"M1 requires same lot_foup: anchor={a_lf!r} vs peer={p_lf!r}"
                )
                return empty

    # M2 stratum lock: chip coord + lot_foup (wafer free)
    if mode == CompareMode.M2_WAFER:
        a_wafer, a_lf = _run_stratum_keys(anchor_rows)
        a_col, a_row = _run_chip_coord(anchor_rows)
        for pid in peers:
            p_wafer, p_lf = _run_stratum_keys(by_run[pid])
            p_col, p_row = _run_chip_coord(by_run[pid])
            if (a_col, a_row) != (p_col, p_row):
                empty.blocked = True
                empty.block_reason = (
                    f"M2 requires same chip coord: "
                    f"anchor=({a_col},{a_row}) vs peer={pid!r} ({p_col},{p_row})"
                )
                return empty
            if a_lf and p_lf and a_lf != p_lf:
                empty.blocked = True
                empty.block_reason = (
                    f"M2 requires same lot_foup: anchor={a_lf!r} vs peer={p_lf!r}"
                )
                return empty
            # Wafer is free; warn via block only if both empty (unusable)
            if not a_wafer and not p_wafer:
                empty.blocked = True
                empty.block_reason = (
                    f"M2 needs wafer_id on both sides (anchor and peer {pid!r})"
                )
                return empty

    # Recipe / lattice gate vs first peer (and all peers)
    for pid in peers:
        ok, reason = recipe_compatible(anchor_rows, by_run[pid])
        if not ok:
            empty.blocked = True
            empty.block_reason = f"recipe_compatible failed vs {pid!r}: {reason}"
            return empty

    anchor_map = _shk_metric_map(anchor_rows, metric_key)
    all_rows: List[Dict[str, Any]] = []
    first_peer = peers[0]
    summary_orphans_a = 0
    summary_orphans_p = 0
    summary_deltas: List[float] = []

    for pid in peers:
        peer_map = _shk_metric_map(by_run[pid], metric_key)
        pair_rows, oa, op = _pair_delta_rows(anchor_map, peer_map, pid)
        _attach_z_scores(pair_rows)
        all_rows.extend(pair_rows)
        if pid == first_peer:
            summary_orphans_a = oa
            summary_orphans_p = op
            summary_deltas = [float(r["delta"]) for r in pair_rows]

    matched_n = len(summary_deltas)
    eligible = matched_n + summary_orphans_a + summary_orphans_p
    missing_rate = (
        float(summary_orphans_a + summary_orphans_p) / float(eligible)
        if eligible > 0
        else 0.0
    )

    mean_d = float(statistics.mean(summary_deltas)) if summary_deltas else 0.0
    median_d = float(statistics.median(summary_deltas)) if summary_deltas else 0.0
    abs_sorted = sorted(abs(d) for d in summary_deltas)
    p95 = _percentile(abs_sorted, 95.0) if abs_sorted else 0.0

    return HomologyResult(
        mode=mode,
        metric=metric_key,
        matched_n=matched_n,
        missing_rate=missing_rate,
        rows=all_rows,
        orphans_anchor=summary_orphans_a,
        orphans_peer=summary_orphans_p,
        mean_delta=mean_d,
        median_delta=median_d,
        p95_abs_delta=p95,
        blocked=False,
        block_reason="",
    )


# ---------------------------------------------------------------------------
# C-P2 — Nested variance stack + Homology Stability Index (HSI)
# ---------------------------------------------------------------------------


def _row_shk_tuple(row: Dict[str, Any]) -> Optional[Tuple[str, int, int, int]]:
    try:
        layer = str(row.get("layer_id") or "")
        fi = int(row.get("fov_index") or 0)
        gr = int(row.get("grid_row"))
        gc = int(row.get("grid_col"))
    except (TypeError, ValueError):
        return None
    if not layer or fi <= 0 or gr < 0 or gc < 0:
        return None
    return layer, fi, gr, gc


def _row_matches_shk(row: Dict[str, Any], shk: SiteHomologyKey) -> bool:
    t = _row_shk_tuple(row)
    if t is None:
        return False
    return t == (shk.layer_id, shk.fov_index, shk.grid_row, shk.grid_col)


def _group_means_var(groups: Dict[Any, List[float]]) -> float:
    """Population variance of group means; 0 if fewer than 2 non-empty groups."""
    means = [statistics.mean(vals) for vals in groups.values() if vals]
    if len(means) < 2:
        return 0.0
    return float(statistics.pvariance(means))


def _pooled_within_var(groups: Dict[Any, List[float]]) -> float:
    """Mean of within-group population variances (groups with n≥2 only)."""
    parts = [
        float(statistics.pvariance(vals))
        for vals in groups.values()
        if len(vals) >= 2
    ]
    if not parts:
        return 0.0
    return float(statistics.mean(parts))


def _stratum_keys(row: Dict[str, Any]) -> Tuple[Any, Any, Any, Any]:
    lot = str(row.get("lot_id") or "")
    foup = (lot, str(row.get("foup_id") or ""))
    wafer = (lot, str(row.get("foup_id") or ""), str(row.get("wafer_id") or ""))
    try:
        cc = int(row.get("chip_col") or 0)
        cr = int(row.get("chip_row") or 0)
    except (TypeError, ValueError):
        cc, cr = 0, 0
    chip = (lot, str(row.get("foup_id") or ""), str(row.get("wafer_id") or ""), cc, cr)
    return lot, foup, wafer, chip


def _variance_components_for_obs(obs: List[Tuple[Dict[str, Any], float]]) -> Dict[str, float]:
    """Successive variance-of-means components for one SHK (or pooled obs)."""
    by_lot: Dict[Any, List[float]] = defaultdict(list)
    by_foup: Dict[Any, List[float]] = defaultdict(list)
    by_wafer: Dict[Any, List[float]] = defaultdict(list)
    by_chip: Dict[Any, List[float]] = defaultdict(list)
    for row, val in obs:
        lot_k, foup_k, wafer_k, chip_k = _stratum_keys(row)
        by_lot[lot_k].append(val)
        by_foup[foup_k].append(val)
        by_wafer[wafer_k].append(val)
        by_chip[chip_k].append(val)

    v_lot = _group_means_var(by_lot)
    v_foup = _group_means_var(by_foup)
    v_wafer = _group_means_var(by_wafer)
    v_chip = _group_means_var(by_chip)
    v_resid = _pooled_within_var(by_chip)

    return {
        "lot": v_lot,
        "foup": max(0.0, v_foup - v_lot),
        "wafer": max(0.0, v_wafer - v_foup),
        "chip": max(0.0, v_chip - v_wafer),
        "residual": v_resid,
    }


def nested_variance(
    table: List[Dict[str, Any]],
    shk_filter: Optional[SiteHomologyKey] = None,
    metric: str = "soh",
) -> Dict[str, Any]:
    """Approximate nested variance stack as lot/foup/wafer/chip/residual %.

    Approximation (C-P2)
    --------------------
    Not a full REML / nested ANOVA. For hierarchy level L we take the
    population variance of group means at L, then set the nested component
    to ``max(0, Var_L − Var_parent)``. Residual is the pooled within-chip
    observation variance. Components are renormalized to percentages that
    sum to ~100.

    When ``shk_filter`` is None, absolute components are computed per SHK
    (sites with ≥2 strata) and averaged, then converted to %.

    Returns
    -------
    dict with keys ``lot``, ``foup``, ``wafer``, ``chip``, ``residual`` (pct),
    plus ``ok`` (bool). On failure also ``reason``. Extra diagnostics:
    ``n_obs``, ``n_shk``, ``method``.
    """
    zeros = {
        "lot": 0.0,
        "foup": 0.0,
        "wafer": 0.0,
        "chip": 0.0,
        "residual": 0.0,
        "ok": False,
        "n_obs": 0,
        "n_shk": 0,
        "method": "variance_of_means_nested",
    }
    metric_key = (metric or "soh").strip().lower()
    rows = list(table or ())

    if shk_filter is not None:
        rows = [r for r in rows if _row_matches_shk(r, shk_filter)]

    # Collect (row, value) with valid metric + SHK
    by_shk: Dict[Tuple[str, int, int, int], List[Tuple[Dict[str, Any], float]]] = defaultdict(list)
    for row in rows:
        shk_t = _row_shk_tuple(row)
        if shk_t is None:
            continue
        val = _metric_value(row, metric_key)
        if val is None:
            continue
        by_shk[shk_t].append((row, float(val)))

    if not by_shk:
        zeros["reason"] = "No usable SHK/metric rows in table"
        return zeros

    # Prefer SHKs that span ≥2 chip strata (meaningful nested split)
    usable: List[List[Tuple[Dict[str, Any], float]]] = []
    for obs in by_shk.values():
        chips = {_stratum_keys(r)[3] for r, _ in obs}
        if len(chips) >= 2 or len(obs) >= 2:
            usable.append(obs)

    if not usable:
        zeros["reason"] = "Insufficient strata (need ≥2 observations or chips per SHK)"
        zeros["n_obs"] = sum(len(v) for v in by_shk.values())
        zeros["n_shk"] = len(by_shk)
        return zeros

    acc = {"lot": 0.0, "foup": 0.0, "wafer": 0.0, "chip": 0.0, "residual": 0.0}
    n_used = 0
    n_obs = 0
    for obs in usable:
        comps = _variance_components_for_obs(obs)
        if sum(comps.values()) <= 0.0:
            continue
        for k in acc:
            acc[k] += comps[k]
        n_used += 1
        n_obs += len(obs)

    if n_used == 0:
        zeros["reason"] = "All candidate SHKs have zero variance across strata"
        zeros["n_obs"] = sum(len(v) for v in by_shk.values())
        zeros["n_shk"] = len(by_shk)
        return zeros

    # Average absolute components across SHKs, then % of total
    for k in acc:
        acc[k] /= float(n_used)
    total = sum(acc.values())
    if total <= 0.0:
        zeros["reason"] = "Total variance is zero after averaging"
        zeros["n_obs"] = n_obs
        zeros["n_shk"] = n_used
        return zeros

    pct = {k: round(100.0 * (acc[k] / total), 2) for k in acc}
    # Fix rounding so percentages sum to 100.00
    drift = round(100.0 - sum(pct.values()), 2)
    if drift != 0.0:
        # Adjust the largest component
        major = max(pct, key=pct.get)
        pct[major] = round(pct[major] + drift, 2)

    return {
        **pct,
        "ok": True,
        "n_obs": n_obs,
        "n_shk": n_used,
        "method": "variance_of_means_nested",
    }


def hsi_rank(
    table: List[Dict[str, Any]],
    metric: str = "soh",
    tol: float = 3.0,
) -> List[Dict[str, Any]]:
    """Rank SHKs by Homology Stability Index (hotspots first).

    ``HSI = 1 − IQR(m_strata) / tol`` per SHK across strata.
    One stratum sample = one chip-instance mean of the metric at that SHK
    (multiple bumps/runs on the same chip are averaged first).

    Each result dict::
        layer_id, fov_index, grid_row, grid_col, hsi, iqr, n_strata, mean_m

    Sorted ascending by ``hsi`` (lowest / least stable first).
    SHKs with ``n_strata < 2`` are omitted.
    """
    metric_key = (metric or "soh").strip().lower()
    tol_v = float(tol) if tol and float(tol) != 0.0 else 1e-9
    # SHK → chip_key → list of metric values
    buckets: Dict[
        Tuple[str, int, int, int], Dict[Any, List[float]]
    ] = defaultdict(lambda: defaultdict(list))

    for row in table or ():
        shk_t = _row_shk_tuple(row)
        if shk_t is None:
            continue
        val = _metric_value(row, metric_key)
        if val is None:
            continue
        chip_k = _stratum_keys(row)[3]
        buckets[shk_t][chip_k].append(float(val))

    out: List[Dict[str, Any]] = []
    for (layer, fi, gr, gc), chip_map in buckets.items():
        strata_means = [statistics.mean(vals) for vals in chip_map.values() if vals]
        n_strata = len(strata_means)
        if n_strata < 2:
            continue
        sorted_m = sorted(strata_means)
        iqr = _percentile(sorted_m, 75.0) - _percentile(sorted_m, 25.0)
        hsi = 1.0 - (iqr / abs(tol_v))
        out.append(
            {
                "layer_id": layer,
                "fov_index": fi,
                "grid_row": gr,
                "grid_col": gc,
                "hsi": float(hsi),
                "iqr": float(iqr),
                "n_strata": n_strata,
                "mean_m": float(statistics.mean(strata_means)),
            }
        )

    out.sort(key=lambda r: (r["hsi"], -r["iqr"], r["layer_id"], r["fov_index"], r["grid_row"], r["grid_col"]))
    return out


# ---------------------------------------------------------------------------
# C-P3 — Pairwise homology matrix + Leave-One-Stratum-Out (LOSO)
# ---------------------------------------------------------------------------


def _chip_unit_label(row: Dict[str, Any]) -> str:
    try:
        cc = int(row.get("chip_col") or 0)
        cr = int(row.get("chip_row") or 0)
    except (TypeError, ValueError):
        cc, cr = 0, 0
    return f"{cc},{cr}"


def _unit_key_for_row(row: Dict[str, Any], mode: CompareMode) -> str:
    """Default unit identity under compare mode (chip / wafer / run)."""
    mode = mode if isinstance(mode, CompareMode) else CompareMode(str(mode))
    if mode == CompareMode.M2_WAFER:
        wid = str(row.get("wafer_id") or "").strip()
        return wid or str(row.get("run_id") or "").strip()
    if mode in (CompareMode.M1_DIE, CompareMode.M5_FOV_BIAS, CompareMode.M6_NESTED):
        # Prefer chip coord; fall back to run_id when chip is (0,0) and run present
        chip = _chip_unit_label(row)
        rid = str(row.get("run_id") or "").strip()
        if chip != "0,0":
            return chip
        return rid or chip
    # M3/M4 etc.: FOUP / lot — use foup or lot when present
    if mode == CompareMode.M3_FOUP:
        return str(row.get("foup_id") or "").strip() or _chip_unit_label(row)
    if mode == CompareMode.M4_LOT:
        return str(row.get("lot_id") or "").strip() or _chip_unit_label(row)
    return str(row.get("run_id") or "").strip() or _chip_unit_label(row)


def _row_matches_unit(row: Dict[str, Any], unit_id: str, mode: CompareMode) -> bool:
    """True if row belongs to ``unit_id`` (run_id, chip ``c,r``, or wafer)."""
    uid = str(unit_id or "").strip()
    if not uid:
        return False
    rid = str(row.get("run_id") or "").strip()
    if rid and rid == uid:
        return True
    if _unit_key_for_row(row, mode) == uid:
        return True
    # Explicit chip "col,row" even when default unit is wafer
    if _chip_unit_label(row) == uid:
        return True
    if str(row.get("wafer_id") or "").strip() == uid:
        return True
    return False


def _unit_shk_metric_map(
    rows: Sequence[Dict[str, Any]],
    metric: str,
) -> Dict[SiteHomologyKey, float]:
    """SHK → mean metric within a unit (duplicates averaged)."""
    buckets: Dict[SiteHomologyKey, List[float]] = defaultdict(list)
    for row in rows:
        shk = shk_from_mes_row(row)
        if shk is None:
            # Homology-table rows may already be flattened
            t = _row_shk_tuple(row)
            if t is None:
                continue
            shk = SiteHomologyKey(
                layer_id=t[0], fov_index=t[1], grid_row=t[2], grid_col=t[3]
            )
        val = _metric_value(row, metric)
        if val is None:
            continue
        buckets[shk].append(float(val))
    return {k: float(statistics.mean(v)) for k, v in buckets.items() if v}


def _median_abs_delta(
    map_a: Dict[SiteHomologyKey, float],
    map_b: Dict[SiteHomologyKey, float],
) -> float:
    """Median |Δ| over matched SHKs; 0 if no overlap."""
    common = set(map_a) & set(map_b)
    if not common:
        return 0.0
    abs_d = [abs(map_a[k] - map_b[k]) for k in common]
    return float(statistics.median(abs_d))


def _discover_unit_labels(
    rows: Sequence[Dict[str, Any]],
    mode: CompareMode,
    unit_ids: Optional[Sequence[str]] = None,
) -> List[str]:
    if unit_ids is not None:
        return [str(u).strip() for u in unit_ids if str(u).strip()]
    seen: List[str] = []
    found: Set[str] = set()
    for row in rows:
        key = _unit_key_for_row(row, mode)
        if not key or key in found:
            continue
        found.add(key)
        seen.append(key)
    # If M1 collapsed to a single chip but multiple runs exist, use run_ids
    if mode == CompareMode.M1_DIE and len(seen) < 2:
        run_seen: List[str] = []
        run_found: Set[str] = set()
        for row in rows:
            rid = str(row.get("run_id") or "").strip()
            if rid and rid not in run_found:
                run_found.add(rid)
                run_seen.append(rid)
        if len(run_seen) >= 2:
            return run_seen
    return seen


def pairwise_homology_matrix(
    rows: List[Dict[str, Any]],
    mode: CompareMode,
    metric: str = "soh",
    unit_ids: Optional[List[str]] = None,
) -> Tuple[List[str], List[List[float]]]:
    """Pairwise median-|Δ| homology distance matrix across units.

    For M1: units default to chips ``(chip_col,chip_row)``; if only one chip
    but multiple ``run_id``s (locked FOV cohort), units fall back to run_ids.
    For M2: units default to ``wafer_id``.

    ``D_ab = median_|Δ|`` over matched SHK between unit a and b.
    Returns ``(labels, matrix NxN)`` with diagonal 0.
    """
    mode = mode if isinstance(mode, CompareMode) else CompareMode(str(mode))
    metric_key = (metric or "soh").strip().lower()
    table = list(rows or ())
    labels = _discover_unit_labels(table, mode, unit_ids)
    n = len(labels)
    matrix: List[List[float]] = [[0.0] * n for _ in range(n)]
    if n == 0:
        return labels, matrix

    unit_maps: List[Dict[SiteHomologyKey, float]] = []
    for lab in labels:
        unit_rows = [r for r in table if _row_matches_unit(r, lab, mode)]
        unit_maps.append(_unit_shk_metric_map(unit_rows, metric_key))

    for i in range(n):
        matrix[i][i] = 0.0
        for j in range(i + 1, n):
            d = _median_abs_delta(unit_maps[i], unit_maps[j])
            matrix[i][j] = d
            matrix[j][i] = d
    return labels, matrix


def _leave_level_key(row: Dict[str, Any], leave_level: str) -> str:
    level = (leave_level or "chip").strip().lower()
    lot = str(row.get("lot_id") or "").strip()
    foup = str(row.get("foup_id") or "").strip()
    wafer = str(row.get("wafer_id") or "").strip()
    try:
        cc = int(row.get("chip_col") or 0)
        cr = int(row.get("chip_row") or 0)
    except (TypeError, ValueError):
        cc, cr = 0, 0
    if level == "lot":
        return lot or "lot:?"
    if level == "foup":
        return f"{lot}|{foup}" if (lot or foup) else "foup:?"
    if level == "wafer":
        return f"{lot}|{foup}|{wafer}" if wafer else (wafer or "wafer:?")
    # chip (default)
    return f"{lot}|{foup}|{wafer}|{cc},{cr}"


def _homologous_stratum_means(
    rows: Sequence[Dict[str, Any]],
    metric: str,
    leave_level: str,
) -> Dict[str, float]:
    """Per-stratum homologous mean = mean of per-SHK means within that stratum."""
    # stratum → SHK → values
    buckets: Dict[str, Dict[Tuple[str, int, int, int], List[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        shk_t = _row_shk_tuple(row)
        if shk_t is None:
            shk = shk_from_mes_row(row)
            if shk is None:
                continue
            shk_t = (shk.layer_id, shk.fov_index, shk.grid_row, shk.grid_col)
        val = _metric_value(row, metric)
        if val is None:
            continue
        sk = _leave_level_key(row, leave_level)
        buckets[sk][shk_t].append(float(val))

    means: Dict[str, float] = {}
    for sk, shk_map in buckets.items():
        shk_means = [statistics.mean(v) for v in shk_map.values() if v]
        if shk_means:
            means[sk] = float(statistics.mean(shk_means))
    return means


def _l_spread(means: Dict[str, float]) -> float:
    """Loss L = sample std of homologous stratum means (0 if <2 strata)."""
    vals = list(means.values())
    if len(vals) < 2:
        return 0.0
    try:
        return float(statistics.stdev(vals))
    except statistics.StatisticsError:
        return 0.0


def loso_impact(
    rows: List[Dict[str, Any]],
    mode: CompareMode,
    leave_level: str = "chip",  # chip|wafer|foup
    metric: str = "soh",
) -> List[Dict[str, Any]]:
    """Leave-one-stratum-out impact ranking.

    ``Impact(s) = L(full) − L(without s)`` where ``L`` is the sample std of
    homologous stratum means (mean-of-SHK-means per stratum instance).

    Returns list sorted by impact descending::
        {level_key, impact, n_obs, L_full, L_without, n_strata_full, mode}

    ``mode`` is recorded for provenance; LOSO pools rows already selected by
    the caller (FOV lock etc. applied upstream).
    """
    mode = mode if isinstance(mode, CompareMode) else CompareMode(str(mode))
    metric_key = (metric or "soh").strip().lower()
    level = (leave_level or "chip").strip().lower()
    if level not in ("chip", "wafer", "foup", "lot"):
        level = "chip"

    table = list(rows or ())
    # Count obs per stratum + build means
    n_obs_by: Dict[str, int] = defaultdict(int)
    usable: List[Dict[str, Any]] = []
    for row in table:
        shk_t = _row_shk_tuple(row)
        if shk_t is None and shk_from_mes_row(row) is None:
            continue
        if _metric_value(row, metric_key) is None:
            continue
        usable.append(row)
        n_obs_by[_leave_level_key(row, level)] += 1

    full_means = _homologous_stratum_means(usable, metric_key, level)
    L_full = _l_spread(full_means)
    keys = sorted(full_means.keys())
    out: List[Dict[str, Any]] = []
    for sk in keys:
        without_rows = [r for r in usable if _leave_level_key(r, level) != sk]
        without_means = _homologous_stratum_means(without_rows, metric_key, level)
        L_wo = _l_spread(without_means)
        out.append(
            {
                "level_key": sk,
                "leave_level": level,
                "impact": float(L_full - L_wo),
                "n_obs": int(n_obs_by.get(sk, 0)),
                "L_full": L_full,
                "L_without": L_wo,
                "n_strata_full": len(full_means),
                "mode": mode.value,
                "metric": metric_key,
            }
        )
    out.sort(key=lambda r: (-float(r["impact"]), str(r["level_key"])))
    return out



def propose_m1_peers(db: Any, anchor_run_id: str) -> List[Dict[str, Any]]:
    """Propose Die↔Die peer FOV runs for M1.

    Same ``wafer_key``, same ``fov_index``, other chips (col/row differ).
    Uses ``InspectionDB.get_run`` / ``list_fov_runs``.

    Each returned dict is a ``fov_runs`` row plus cheap coverage hints:
      ``n_objects``, ``coverage_hint`` (peer n_objects / max(anchor n_objects, 1)).
    """
    rid = str(anchor_run_id or "").strip()
    if not rid or db is None:
        return []
    get_run = getattr(db, "get_run", None)
    list_fov_runs = getattr(db, "list_fov_runs", None)
    if not callable(get_run) or not callable(list_fov_runs):
        return []

    anchor = get_run(rid)
    if not anchor:
        return []

    wafer_key = str(anchor.get("wafer_key") or "").strip()
    if not wafer_key:
        return []

    try:
        fov_index = int(anchor.get("fov_index") or 0)
    except (TypeError, ValueError):
        fov_index = 0
    if fov_index <= 0:
        return []

    try:
        a_col = int(anchor.get("chip_col") or 0)
    except (TypeError, ValueError):
        a_col = 0
    try:
        a_row = int(anchor.get("chip_row") or 0)
    except (TypeError, ValueError):
        a_row = 0

    try:
        a_n = int(anchor.get("n_objects") or 0)
    except (TypeError, ValueError):
        a_n = 0
    if a_n <= 0:
        mes = anchor.get("mes_objects") or []
        a_n = len(mes) if isinstance(mes, list) else 0

    candidates = list_fov_runs(wafer_key=wafer_key, limit=2000)
    out: List[Dict[str, Any]] = []
    for run in candidates or ():
        peer_id = str(run.get("run_id") or "").strip()
        if not peer_id or peer_id == rid:
            continue
        try:
            p_fov = int(run.get("fov_index") or 0)
        except (TypeError, ValueError):
            p_fov = 0
        if p_fov != fov_index:
            continue
        try:
            p_col = int(run.get("chip_col") or 0)
        except (TypeError, ValueError):
            p_col = 0
        try:
            p_row = int(run.get("chip_row") or 0)
        except (TypeError, ValueError):
            p_row = 0
        if p_col == a_col and p_row == a_row:
            continue  # same chip (retest) — not an M1 peer
        try:
            p_n = int(run.get("n_objects") or 0)
        except (TypeError, ValueError):
            p_n = 0
        d = dict(run)
        d["n_objects"] = p_n
        d["coverage_hint"] = float(p_n) / float(max(a_n, 1))
        out.append(d)
    return out
