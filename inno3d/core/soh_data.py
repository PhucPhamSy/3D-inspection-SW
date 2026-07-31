"""
SOH analysis data helpers — bridge inspection DB / CSV into SampleData.

Used by ``inno3d.tabs.analysis``.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from inno3d.core.inspection_db import InspectionDB


def mes_row_to_analysis_dict(obj: Dict[str, Any]) -> dict:
    """Map a joined mes_objects row to the CSV-compatible dict used by analysis engines."""
    layer = str(obj.get("layer_name") or "Default").strip() or "Default"
    soh = obj.get("soh")
    ratio = obj.get("ratio")
    try:
        soh_f = float(soh) if soh is not None else 0.0
    except (TypeError, ValueError):
        soh_f = 0.0
    try:
        ratio_f = float(ratio) if ratio is not None else 0.0
    except (TypeError, ValueError):
        ratio_f = 0.0
    # Analysis code treats ratio > 1 as percent
    ratio_out = ratio_f * 100.0 if 0.0 <= ratio_f <= 1.0 else ratio_f

    c1 = obj.get("c1_volume")
    c2 = obj.get("c2_volume")
    try:
        c1_f = float(c1) if c1 is not None else 0.0
    except (TypeError, ValueError):
        c1_f = 0.0
    try:
        c2_f = float(c2) if c2 is not None else 0.0
    except (TypeError, ValueError):
        c2_f = 0.0

    gr = obj.get("grid_row")
    gc = obj.get("grid_col")
    try:
        gr_i = int(gr) if gr is not None else 0
    except (TypeError, ValueError):
        gr_i = 0
    try:
        gc_i = int(gc) if gc is not None else 0
    except (TypeError, ValueError):
        gc_i = 0

    grid_label = f"({gr_i},{gc_i})"
    judgment = str(obj.get("judgment") or "OK").strip() or "OK"

    return {
        "Layer": layer,
        "_layer_resolved": layer,
        "SOH (um)": f"{soh_f}",
        "SOH(µm)": f"{soh_f}",
        "Ratio": f"{ratio_out}",
        "Ratio (Void/(TGV+Void))": f"{ratio_out}",
        "Bump Volume (um3)": f"{c1_f}",
        "Bump Vol(um3)": f"{c1_f}",
        "Void Volume (um3)": f"{c2_f}",
        "Void Vol(um3)": f"{c2_f}",
        "Judgment": judgment,
        "judgment": judgment,
        "Grid(R,C)": grid_label,
        "grid_row": str(gr_i),
        "grid_col": str(gc_i),
        "#": str(obj.get("row_id") or ""),
        "Centroid_X": str(obj.get("centroid_x") or 0),
        "Centroid_Y": str(obj.get("centroid_y") or 0),
        "Centroid_Z": str(obj.get("centroid_z") or 0),
        "centroid_x": str(obj.get("centroid_x") or 0),
        "centroid_y": str(obj.get("centroid_y") or 0),
        "centroid_z": str(obj.get("centroid_z") or 0),
        "Z_min": str(obj.get("z_min") or 0),
        "Z_max": str(obj.get("z_max") or 0),
        "_run_id": str(obj.get("run_id") or ""),
        "_chip_col": str(obj.get("chip_col") or 0),
        "_chip_row": str(obj.get("chip_row") or 0),
        "_fov_index": str(obj.get("fov_index") or 0),
        "_wafer_id": str(obj.get("wafer_id") or ""),
    }


def sample_name_from_run(run: Dict[str, Any]) -> str:
    wid = str(run.get("wafer_id") or "W")
    c = int(run.get("chip_col") or 0)
    r = int(run.get("chip_row") or 0)
    fi = int(run.get("fov_index") or 0)
    return f"{wid}_C{c}_{r}_P{fi}"


@dataclass
class SampleData:
    """Parsed measurement data for one analysis unit (CSV folder or FOV run)."""

    name: str
    path: str
    layers: Dict[str, Any] = field(default_factory=dict)
    raw_rows: List[dict] = field(default_factory=list)
    # Production / DB meta
    run_id: str = ""
    wafer_key: str = ""
    wafer_id: str = ""
    lot_foup_id: str = ""
    chip_col: int = 0
    chip_row: int = 0
    fov_index: int = 0
    judgment: str = ""
    source: str = "csv"  # "csv" | "db"


def build_samples_from_mes(
    mes_rows: Sequence[Dict[str, Any]],
    runs_by_id: Dict[str, Dict[str, Any]],
    *,
    compute_layer_stats_fn,
    ng_threshold: float,
) -> List[SampleData]:
    """Group joined MES rows by run_id → SampleData list."""
    by_run: Dict[str, List[dict]] = defaultdict(list)
    for obj in mes_rows:
        rid = str(obj.get("run_id") or "")
        if not rid:
            continue
        by_run[rid].append(mes_row_to_analysis_dict(obj))

    samples: List[SampleData] = []
    for rid, rows in by_run.items():
        run = runs_by_id.get(rid) or {}
        # Fill run meta from first mes row if needed
        if not run and mes_rows:
            for m in mes_rows:
                if str(m.get("run_id")) == rid:
                    run = {
                        "run_id": rid,
                        "wafer_key": m.get("wafer_key", ""),
                        "wafer_id": m.get("wafer_id", ""),
                        "lot_foup_id": m.get("lot_foup_id", ""),
                        "chip_col": m.get("chip_col", 0),
                        "chip_row": m.get("chip_row", 0),
                        "fov_index": m.get("fov_index", 0),
                        "results_dir": m.get("results_dir", ""),
                        "judgment": m.get("run_judgment", ""),
                    }
                    break

        layer_groups: Dict[str, List[dict]] = defaultdict(list)
        for row in rows:
            layer_groups[row.get("_layer_resolved", "Default")].append(row)

        layers = {
            ln: compute_layer_stats_fn(rs, ln, ng_threshold)
            for ln, rs in layer_groups.items()
        }
        name = sample_name_from_run(run) if run else rid
        samples.append(
            SampleData(
                name=name,
                path=str(run.get("results_dir") or run.get("input_path") or ""),
                layers=layers,
                raw_rows=rows,
                run_id=rid,
                wafer_key=str(run.get("wafer_key") or ""),
                wafer_id=str(run.get("wafer_id") or ""),
                lot_foup_id=str(run.get("lot_foup_id") or ""),
                chip_col=int(run.get("chip_col") or 0),
                chip_row=int(run.get("chip_row") or 0),
                fov_index=int(run.get("fov_index") or 0),
                judgment=str(run.get("judgment") or ""),
                source="db",
            )
        )

    samples.sort(
        key=lambda s: (s.wafer_id, s.chip_row, s.chip_col, s.fov_index, s.name)
    )
    return samples


def load_samples_from_db(
    db: InspectionDB,
    *,
    wafer_key: str = "",
    run_ids: Optional[Iterable[str]] = None,
    ng_threshold: float = 0.05,
    compute_layer_stats_fn=None,
) -> List[SampleData]:
    """Load FOV runs as SampleData from inspection DB."""
    if compute_layer_stats_fn is None:
        raise ValueError("compute_layer_stats_fn required")

    ids = [str(x) for x in (run_ids or []) if x] if run_ids is not None else None
    if ids is not None and not ids:
        return []

    # Resolve run list
    if ids is not None:
        runs = []
        for rid in ids:
            r = db.get_run(rid)
            if r:
                # strip heavy nested lists from cache map
                runs.append({k: v for k, v in r.items() if k not in ("mes_objects", "artifacts", "timeline")})
        # Prefer joined mes query for speed
        mes = db.list_mes_joined(run_ids=ids)
        runs_by_id = {str(r["run_id"]): r for r in runs}
        # Ensure all runs present from mes meta
        for m in mes:
            rid = str(m.get("run_id") or "")
            if rid and rid not in runs_by_id:
                runs_by_id[rid] = {
                    "run_id": rid,
                    "wafer_key": m.get("wafer_key", ""),
                    "wafer_id": m.get("wafer_id", ""),
                    "lot_foup_id": m.get("lot_foup_id", ""),
                    "chip_col": m.get("chip_col", 0),
                    "chip_row": m.get("chip_row", 0),
                    "fov_index": m.get("fov_index", 0),
                    "results_dir": m.get("results_dir", ""),
                    "judgment": m.get("run_judgment", ""),
                }
        return build_samples_from_mes(
            mes, runs_by_id, compute_layer_stats_fn=compute_layer_stats_fn, ng_threshold=ng_threshold
        )

    if not wafer_key:
        return []

    runs = db.list_fov_runs(wafer_key=wafer_key, limit=5000)
    runs_by_id = {str(r["run_id"]): r for r in runs}
    mes = db.list_mes_joined(wafer_key=wafer_key)
    return build_samples_from_mes(
        mes, runs_by_id, compute_layer_stats_fn=compute_layer_stats_fn, ng_threshold=ng_threshold
    )


def aggregate_layers_from_samples(
    samples: Sequence[SampleData],
    ng_threshold: float,
    compute_layer_stats_fn,
) -> Dict[str, Any]:
    """Merge all samples' raw rows and compute per-layer stats (scope-level)."""
    layer_groups: Dict[str, List[dict]] = defaultdict(list)
    for s in samples:
        for row in s.raw_rows:
            ln = row.get("_layer_resolved", row.get("Layer", "Default"))
            layer_groups[str(ln)].append(row)
    return {
        ln: compute_layer_stats_fn(rows, ln, ng_threshold)
        for ln, rows in layer_groups.items()
    }


def aggregate_fov_from_samples(samples: Sequence[SampleData]) -> List[Dict[str, Any]]:
    """Mean SOH / NG rate per FOV index from in-memory samples."""
    buckets: Dict[int, List[Tuple[float, float]]] = defaultdict(list)
    # (soh, ratio) per object
    for s in samples:
        fi = int(s.fov_index or 0)
        if fi <= 0:
            continue
        for row in s.raw_rows:
            try:
                soh = float(row.get("SOH (um)") or row.get("SOH(µm)") or 0)
            except (TypeError, ValueError):
                soh = 0.0
            try:
                ratio = float(str(row.get("Ratio") or "0").replace("%", ""))
                if ratio > 1.0:
                    ratio = ratio / 100.0
            except (TypeError, ValueError):
                ratio = 0.0
            buckets[fi].append((soh, ratio))

    out: List[Dict[str, Any]] = []
    for fi in sorted(buckets.keys()):
        vals = buckets[fi]
        sohs = [v[0] for v in vals]
        ratios = [v[1] for v in vals]
        n = len(vals)
        n_ng = sum(1 for r in ratios if r >= 0.05)
        mean_soh = sum(sohs) / n if n else 0.0
        out.append(
            {
                "key": fi,
                "n": n,
                "mean_soh": mean_soh,
                "ng_rate": (n_ng / n) if n else 0.0,
                "mean_ratio": (sum(ratios) / n) if n else 0.0,
                "n_runs": len({s.run_id for s in samples if int(s.fov_index or 0) == fi}),
            }
        )
    return out


def _aggregate_metric_value(row: Dict[str, Any]) -> float:
    """Return the map metric value from a SQL aggregate row.

    Callers may add ``metric`` (or ``value``) to select a non-SOH aggregate
    before passing rows to the grid helpers.  The natural default remains
    ``mean_soh`` for the SOH map.
    """
    for key in ("metric", "value", "mean_soh"):
        try:
            value = row.get(key)
            if value is not None:
                return float(value)
        except (TypeError, ValueError):
            continue
    return 0.0


def chip_metric_grid(agg_rows: List[Dict[str, Any]]) -> Dict[Tuple[int, int], float]:
    """Convert chip SQL aggregate rows to ``(chip_col, chip_row) → metric``."""
    grid: Dict[Tuple[int, int], float] = {}
    for row in agg_rows:
        try:
            col = int(row.get("chip_col"))
            row_idx = int(row.get("chip_row"))
        except (TypeError, ValueError):
            continue
        grid[(col, row_idx)] = _aggregate_metric_value(row)
    return grid


def fov_metric_grid(agg_rows: List[Dict[str, Any]]) -> Dict[int, float]:
    """Convert FOV SQL aggregate rows to ``P1..P9 → metric``."""
    grid: Dict[int, float] = {}
    for row in agg_rows:
        try:
            fov_index = int(row.get("key"))
        except (TypeError, ValueError):
            continue
        if 1 <= fov_index <= 9:
            grid[fov_index] = _aggregate_metric_value(row)
    return grid
