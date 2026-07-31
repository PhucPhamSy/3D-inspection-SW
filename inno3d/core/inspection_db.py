"""
Inspection production database (SQLite).

Stores metadata + KPI + artifact paths for Online / offline review.
Heavy files (volumes, masks) stay on disk under Results/.

Default location (dev & frozen)::

    <app_dir>/Inno3D_Data/inspection.db
"""
from __future__ import annotations

import json
import math
import os
import sqlite3
import sys
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _std(vals: List[float]) -> float:
    """Sample standard deviation (n-1); 0 if n < 2."""
    n = len(vals)
    if n < 2:
        return 0.0
    mean = sum(vals) / n
    var = sum((v - mean) ** 2 for v in vals) / (n - 1)
    return math.sqrt(var)


_SCHEMA_VERSION = 2  # v2: dropped RecipeID (path + DB columns)
_lock = threading.RLock()
_default_db: Optional["InspectionDB"] = None


def make_wafer_key(
    date_folder: str = "",
    lot_foup_id: str = "",
    wafer_id: str = "",
) -> str:
    """Canonical wafer identity: date | LotID_FoupID | WaferID (no RecipeID)."""
    return "|".join(
        [
            date_folder or "",
            lot_foup_id or "",
            wafer_id or "",
        ]
    )


def normalize_wafer_key(key: str) -> str:
    """Strip legacy 4th RecipeID segment from wafer_key if present."""
    parts = [p for p in str(key or "").split("|")]
    if len(parts) >= 3:
        return "|".join(parts[:3])
    return str(key or "")


def app_data_dir() -> Path:
    if getattr(sys, "frozen", False):
        base = Path(sys.executable).resolve().parent
    else:
        base = Path(__file__).resolve().parents[2]
    d = base / "Inno3D_Data"
    d.mkdir(parents=True, exist_ok=True)
    return d


def default_db_path() -> Path:
    return app_data_dir() / "inspection.db"


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _new_id(prefix: str = "") -> str:
    u = uuid.uuid4().hex[:12]
    return f"{prefix}{u}" if prefix else u


class InspectionDB:
    """Thread-safe thin wrapper around SQLite inspection catalog."""

    def __init__(self, db_path: Optional[os.PathLike] = None):
        self.db_path = Path(db_path) if db_path else default_db_path()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def _conn(self):
        with _lock:
            con = sqlite3.connect(str(self.db_path), timeout=30.0)
            con.row_factory = sqlite3.Row
            con.execute("PRAGMA foreign_keys = ON")
            try:
                yield con
                con.commit()
            except Exception:
                con.rollback()
                raise
            finally:
                con.close()

    def _init_schema(self):
        with self._conn() as con:
            con.executescript(
                """
                CREATE TABLE IF NOT EXISTS meta (
                    key TEXT PRIMARY KEY,
                    value TEXT
                );

                CREATE TABLE IF NOT EXISTS wafers (
                    wafer_key TEXT PRIMARY KEY,
                    date_folder TEXT,
                    lot_foup_id TEXT,
                    lot_id TEXT,
                    foup_id TEXT,
                    wafer_id TEXT,
                    grid_cols INTEGER DEFAULT 0,
                    grid_rows INTEGER DEFAULT 0,
                    first_seen TEXT,
                    last_seen TEXT,
                    notes TEXT
                );

                CREATE TABLE IF NOT EXISTS chips (
                    chip_key TEXT PRIMARY KEY,
                    wafer_key TEXT NOT NULL,
                    chip_col INTEGER,
                    chip_row INTEGER,
                    chip_folder TEXT,
                    final_bin INTEGER DEFAULT 2,
                    FOREIGN KEY (wafer_key) REFERENCES wafers(wafer_key)
                );

                CREATE TABLE IF NOT EXISTS fov_runs (
                    run_id TEXT PRIMARY KEY,
                    wafer_key TEXT,
                    chip_key TEXT,
                    date_folder TEXT,
                    lot_foup_id TEXT,
                    wafer_id TEXT,
                    chip_col INTEGER,
                    chip_row INTEGER,
                    fov_index INTEGER,
                    fov_folder TEXT,
                    input_path TEXT,
                    results_dir TEXT,
                    judgment TEXT,
                    final_bin INTEGER,
                    n_objects INTEGER DEFAULT 0,
                    n_ng INTEGER DEFAULT 0,
                    n_ok INTEGER DEFAULT 0,
                    enhance_sec REAL,
                    seg_sec REAL,
                    mes_sec REAL,
                    b2b_sec REAL,
                    total_sec REAL,
                    enhance_provider TEXT,
                    config_path TEXT,
                    dll_dir TEXT,
                    recipe_json TEXT,
                    host_path TEXT,
                    started_at TEXT,
                    finished_at TEXT,
                    status TEXT DEFAULT 'ok',
                    error_message TEXT,
                    FOREIGN KEY (wafer_key) REFERENCES wafers(wafer_key)
                );

                CREATE TABLE IF NOT EXISTS mes_objects (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    row_id INTEGER,
                    layer_name TEXT,
                    grid_row INTEGER,
                    grid_col INTEGER,
                    soh REAL,
                    c1_volume REAL,
                    c2_volume REAL,
                    ratio REAL,
                    judgment TEXT,
                    pitch_x REAL,
                    pitch_y REAL,
                    z_min INTEGER,
                    z_max INTEGER,
                    y_min INTEGER,
                    y_max INTEGER,
                    x_min INTEGER,
                    x_max INTEGER,
                    centroid_z REAL,
                    centroid_y REAL,
                    centroid_x REAL,
                    extra_json TEXT,
                    FOREIGN KEY (run_id) REFERENCES fov_runs(run_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS artifacts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    path TEXT NOT NULL,
                    exists_flag INTEGER DEFAULT 1,
                    FOREIGN KEY (run_id) REFERENCES fov_runs(run_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS timeline (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    ts TEXT,
                    step TEXT,
                    message TEXT,
                    duration_sec REAL,
                    FOREIGN KEY (run_id) REFERENCES fov_runs(run_id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_runs_wafer ON fov_runs(wafer_key);
                CREATE INDEX IF NOT EXISTS idx_runs_lot ON fov_runs(lot_foup_id);
                CREATE INDEX IF NOT EXISTS idx_runs_finished ON fov_runs(finished_at);
                CREATE INDEX IF NOT EXISTS idx_mes_run ON mes_objects(run_id);
                CREATE INDEX IF NOT EXISTS idx_mes_layer_grid
                  ON mes_objects(layer_name, grid_row, grid_col);
                CREATE INDEX IF NOT EXISTS idx_art_run ON artifacts(run_id);
                """
            )
            row = con.execute(
                "SELECT value FROM meta WHERE key='schema_version'"
            ).fetchone()
            try:
                current = int(row["value"]) if row else 0
            except (TypeError, ValueError, KeyError):
                current = 0

            # Existing DB created before v2 still has recipe_id columns
            # (CREATE IF NOT EXISTS does not rewrite tables).
            if self._table_has_column(con, "wafers", "recipe_id") or self._table_has_column(
                con, "fov_runs", "recipe_id"
            ):
                self._migrate_drop_recipe_id(con)
                current = 0  # force version stamp below

            if not row:
                con.execute(
                    "INSERT INTO meta(key, value) VALUES('schema_version', ?)",
                    (str(_SCHEMA_VERSION),),
                )
            elif current < _SCHEMA_VERSION:
                con.execute(
                    "UPDATE meta SET value=? WHERE key='schema_version'",
                    (str(_SCHEMA_VERSION),),
                )

    @staticmethod
    def _table_has_column(con: sqlite3.Connection, table: str, column: str) -> bool:
        try:
            rows = con.execute(f"PRAGMA table_info({table})").fetchall()
        except sqlite3.Error:
            return False
        for r in rows:
            # sqlite3.Row or tuple: name is index 1
            name = r[1] if not isinstance(r, sqlite3.Row) else r["name"]
            if str(name) == column:
                return True
        return False

    def _migrate_drop_recipe_id(self, con: sqlite3.Connection) -> None:
        """Rebuild wafers/fov_runs without recipe_id; normalize wafer_key to 3 segments."""
        con.execute("PRAGMA foreign_keys=OFF")
        try:
            # ---- wafers ----
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS wafers_v2 (
                    wafer_key TEXT PRIMARY KEY,
                    date_folder TEXT,
                    lot_foup_id TEXT,
                    lot_id TEXT,
                    foup_id TEXT,
                    wafer_id TEXT,
                    grid_cols INTEGER DEFAULT 0,
                    grid_rows INTEGER DEFAULT 0,
                    first_seen TEXT,
                    last_seen TEXT,
                    notes TEXT
                )
                """
            )
            con.execute("DELETE FROM wafers_v2")
            key_map: Dict[str, str] = {}
            for row in con.execute("SELECT * FROM wafers").fetchall():
                d = dict(row)
                old_key = str(d.get("wafer_key") or "")
                new_key = make_wafer_key(
                    d.get("date_folder", "") or "",
                    d.get("lot_foup_id", "") or "",
                    d.get("wafer_id", "") or "",
                )
                if not new_key or new_key == "||":
                    new_key = normalize_wafer_key(old_key)
                key_map[old_key] = new_key
                existing = con.execute(
                    "SELECT wafer_key, first_seen, last_seen, grid_cols, grid_rows "
                    "FROM wafers_v2 WHERE wafer_key=?",
                    (new_key,),
                ).fetchone()
                if existing:
                    # Merge duplicate recipes under same wafer identity
                    fs = d.get("first_seen") or existing["first_seen"]
                    ls = d.get("last_seen") or existing["last_seen"]
                    if existing["first_seen"] and d.get("first_seen"):
                        fs = min(str(existing["first_seen"]), str(d["first_seen"]))
                    if existing["last_seen"] and d.get("last_seen"):
                        ls = max(str(existing["last_seen"]), str(d["last_seen"]))
                    gc = max(int(existing["grid_cols"] or 0), int(d.get("grid_cols") or 0))
                    gr = max(int(existing["grid_rows"] or 0), int(d.get("grid_rows") or 0))
                    con.execute(
                        "UPDATE wafers_v2 SET first_seen=?, last_seen=?, "
                        "grid_cols=?, grid_rows=? WHERE wafer_key=?",
                        (fs, ls, gc, gr, new_key),
                    )
                else:
                    con.execute(
                        """
                        INSERT INTO wafers_v2(
                            wafer_key, date_folder, lot_foup_id, lot_id, foup_id,
                            wafer_id, grid_cols, grid_rows, first_seen, last_seen, notes
                        ) VALUES (?,?,?,?,?,?,?,?,?,?,?)
                        """,
                        (
                            new_key,
                            d.get("date_folder") or "",
                            d.get("lot_foup_id") or "",
                            d.get("lot_id") or "",
                            d.get("foup_id") or "",
                            d.get("wafer_id") or "",
                            int(d.get("grid_cols") or 0),
                            int(d.get("grid_rows") or 0),
                            d.get("first_seen"),
                            d.get("last_seen"),
                            d.get("notes"),
                        ),
                    )

            # ---- chips (remap wafer_key + chip_key) ----
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS chips_v2 (
                    chip_key TEXT PRIMARY KEY,
                    wafer_key TEXT NOT NULL,
                    chip_col INTEGER,
                    chip_row INTEGER,
                    chip_folder TEXT,
                    final_bin INTEGER DEFAULT 2
                )
                """
            )
            con.execute("DELETE FROM chips_v2")
            for row in con.execute("SELECT * FROM chips").fetchall():
                d = dict(row)
                old_wk = str(d.get("wafer_key") or "")
                new_wk = key_map.get(old_wk) or normalize_wafer_key(old_wk)
                col = int(d.get("chip_col") or 0)
                row_i = int(d.get("chip_row") or 0)
                new_ck = f"{new_wk}|{col}|{row_i}"
                existing = con.execute(
                    "SELECT chip_key, final_bin FROM chips_v2 WHERE chip_key=?",
                    (new_ck,),
                ).fetchone()
                if existing:
                    # Prefer NG (1) over OK (8) over Pending (2)
                    old_fb = int(existing["final_bin"] or 2)
                    new_fb = int(d.get("final_bin") or 2)
                    if new_fb == 1 or (old_fb != 1 and new_fb == 8):
                        fb = new_fb
                    else:
                        fb = old_fb
                    con.execute(
                        "UPDATE chips_v2 SET final_bin=?, chip_folder=? WHERE chip_key=?",
                        (fb, d.get("chip_folder") or "", new_ck),
                    )
                else:
                    con.execute(
                        """
                        INSERT INTO chips_v2(
                            chip_key, wafer_key, chip_col, chip_row, chip_folder, final_bin
                        ) VALUES (?,?,?,?,?,?)
                        """,
                        (
                            new_ck,
                            new_wk,
                            col,
                            row_i,
                            d.get("chip_folder") or "",
                            int(d.get("final_bin") or 2),
                        ),
                    )

            # ---- fov_runs without recipe_id ----
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS fov_runs_v2 (
                    run_id TEXT PRIMARY KEY,
                    wafer_key TEXT,
                    chip_key TEXT,
                    date_folder TEXT,
                    lot_foup_id TEXT,
                    wafer_id TEXT,
                    chip_col INTEGER,
                    chip_row INTEGER,
                    fov_index INTEGER,
                    fov_folder TEXT,
                    input_path TEXT,
                    results_dir TEXT,
                    judgment TEXT,
                    final_bin INTEGER,
                    n_objects INTEGER DEFAULT 0,
                    n_ng INTEGER DEFAULT 0,
                    n_ok INTEGER DEFAULT 0,
                    enhance_sec REAL,
                    seg_sec REAL,
                    mes_sec REAL,
                    b2b_sec REAL,
                    total_sec REAL,
                    enhance_provider TEXT,
                    config_path TEXT,
                    dll_dir TEXT,
                    recipe_json TEXT,
                    host_path TEXT,
                    started_at TEXT,
                    finished_at TEXT,
                    status TEXT DEFAULT 'ok',
                    error_message TEXT
                )
                """
            )
            con.execute("DELETE FROM fov_runs_v2")
            for row in con.execute("SELECT * FROM fov_runs").fetchall():
                d = dict(row)
                old_wk = str(d.get("wafer_key") or "")
                new_wk = key_map.get(old_wk) or normalize_wafer_key(old_wk)
                col = int(d.get("chip_col") or 0)
                row_i = int(d.get("chip_row") or 0)
                new_ck = f"{new_wk}|{col}|{row_i}"
                con.execute(
                    """
                    INSERT OR REPLACE INTO fov_runs_v2(
                        run_id, wafer_key, chip_key, date_folder, lot_foup_id, wafer_id,
                        chip_col, chip_row, fov_index, fov_folder, input_path, results_dir,
                        judgment, final_bin, n_objects, n_ng, n_ok,
                        enhance_sec, seg_sec, mes_sec, b2b_sec, total_sec,
                        enhance_provider, config_path, dll_dir, recipe_json, host_path,
                        started_at, finished_at, status, error_message
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        d.get("run_id"),
                        new_wk,
                        new_ck,
                        d.get("date_folder") or "",
                        d.get("lot_foup_id") or "",
                        d.get("wafer_id") or "",
                        col,
                        row_i,
                        int(d.get("fov_index") or 0),
                        d.get("fov_folder") or "",
                        d.get("input_path") or "",
                        d.get("results_dir") or "",
                        d.get("judgment") or "",
                        d.get("final_bin"),
                        int(d.get("n_objects") or 0),
                        int(d.get("n_ng") or 0),
                        int(d.get("n_ok") or 0),
                        d.get("enhance_sec"),
                        d.get("seg_sec"),
                        d.get("mes_sec"),
                        d.get("b2b_sec"),
                        d.get("total_sec"),
                        d.get("enhance_provider") or "",
                        d.get("config_path") or "",
                        d.get("dll_dir") or "",
                        d.get("recipe_json") or "",
                        d.get("host_path") or "",
                        d.get("started_at"),
                        d.get("finished_at"),
                        d.get("status") or "ok",
                        d.get("error_message") or "",
                    ),
                )

            con.executescript(
                """
                DROP TABLE IF EXISTS wafers;
                DROP TABLE IF EXISTS chips;
                DROP TABLE IF EXISTS fov_runs;
                ALTER TABLE wafers_v2 RENAME TO wafers;
                ALTER TABLE chips_v2 RENAME TO chips;
                ALTER TABLE fov_runs_v2 RENAME TO fov_runs;
                """
            )
        finally:
            con.execute("PRAGMA foreign_keys=ON")

    # ------------------------------------------------------------------
    # Write API
    # ------------------------------------------------------------------
    def upsert_wafer(
        self,
        *,
        date_folder: str = "",
        lot_foup_id: str = "",
        lot_id: str = "",
        foup_id: str = "",
        wafer_id: str = "",
        grid_cols: int = 0,
        grid_rows: int = 0,
    ) -> str:
        # Identity: date / LotID_FoupID / WaferID  (no RecipeID)
        wafer_key = make_wafer_key(date_folder, lot_foup_id, wafer_id)
        now = _utc_now()
        with self._conn() as con:
            existing = con.execute(
                "SELECT wafer_key FROM wafers WHERE wafer_key=?", (wafer_key,)
            ).fetchone()
            if existing:
                con.execute(
                    """
                    UPDATE wafers SET last_seen=?,
                        grid_cols=CASE WHEN ? > 0 THEN ? ELSE grid_cols END,
                        grid_rows=CASE WHEN ? > 0 THEN ? ELSE grid_rows END
                    WHERE wafer_key=?
                    """,
                    (now, grid_cols, grid_cols, grid_rows, grid_rows, wafer_key),
                )
            else:
                con.execute(
                    """
                    INSERT INTO wafers(
                        wafer_key, date_folder, lot_foup_id, lot_id, foup_id,
                        wafer_id, grid_cols, grid_rows, first_seen, last_seen
                    ) VALUES (?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        wafer_key,
                        date_folder,
                        lot_foup_id,
                        lot_id,
                        foup_id,
                        wafer_id,
                        grid_cols,
                        grid_rows,
                        now,
                        now,
                    ),
                )
        return wafer_key

    def upsert_chip(
        self,
        wafer_key: str,
        chip_col: int,
        chip_row: int,
        chip_folder: str = "",
        final_bin: int = 2,
    ) -> str:
        chip_key = f"{wafer_key}|{chip_col}|{chip_row}"
        with self._conn() as con:
            existing = con.execute(
                "SELECT chip_key FROM chips WHERE chip_key=?", (chip_key,)
            ).fetchone()
            if existing:
                con.execute(
                    "UPDATE chips SET final_bin=?, chip_folder=? WHERE chip_key=?",
                    (final_bin, chip_folder, chip_key),
                )
            else:
                con.execute(
                    """
                    INSERT INTO chips(chip_key, wafer_key, chip_col, chip_row, chip_folder, final_bin)
                    VALUES (?,?,?,?,?,?)
                    """,
                    (chip_key, wafer_key, chip_col, chip_row, chip_folder, final_bin),
                )
        return chip_key

    def record_fov_run(self, payload: Dict[str, Any]) -> str:
        """Insert one completed FOV inspection.

        Expected keys (all optional except useful paths)::

            host_path, input_path, results_dir,
            date_folder, lot_foup_id, lot_id, foup_id, wafer_id,
            chip_col, chip_row, chip_folder, fov_index, fov_folder,
            judgment ('OK'|'NG'|'ERROR'|'PENDING'), final_bin,
            n_objects, n_ng, n_ok,
            enhance_sec, seg_sec, mes_sec, b2b_sec, total_sec,
            enhance_provider, config_path, dll_dir, recipe_json,
            started_at, finished_at, status, error_message,
            mes_objects: list[dict],
            artifacts: list[{kind, path}],
            timeline: list[{ts, step, message, duration_sec}]
        """
        run_id = payload.get("run_id") or _new_id("run_")
        finished = payload.get("finished_at") or _utc_now()
        started = payload.get("started_at") or finished

        wafer_key = self.upsert_wafer(
            date_folder=payload.get("date_folder", "") or "",
            lot_foup_id=payload.get("lot_foup_id", "") or "",
            lot_id=payload.get("lot_id", "") or "",
            foup_id=payload.get("foup_id", "") or "",
            wafer_id=payload.get("wafer_id", "") or "",
            grid_cols=int(payload.get("grid_cols") or 0),
            grid_rows=int(payload.get("grid_rows") or 0),
        )
        # chips.final_bin is CHIP rollup (Pending/OK/NG), not the FOV bin.
        # Prefer explicit chip_final_bin from Online context; never use FOV final_bin alone
        # (that previously painted the die green after a single FOV OK).
        try:
            chip_final = int(
                payload["chip_final_bin"]
                if payload.get("chip_final_bin") is not None
                else 2
            )
        except (TypeError, ValueError, KeyError):
            chip_final = 2
        chip_key = self.upsert_chip(
            wafer_key,
            int(payload.get("chip_col") or 0),
            int(payload.get("chip_row") or 0),
            chip_folder=payload.get("chip_folder", "") or "",
            final_bin=chip_final,
        )

        recipe_json = payload.get("recipe_json")
        if recipe_json is not None and not isinstance(recipe_json, str):
            recipe_json = json.dumps(recipe_json, ensure_ascii=False)

        with self._conn() as con:
            con.execute(
                """
                INSERT OR REPLACE INTO fov_runs(
                    run_id, wafer_key, chip_key, date_folder, lot_foup_id, wafer_id,
                    chip_col, chip_row, fov_index, fov_folder, input_path, results_dir,
                    judgment, final_bin, n_objects, n_ng, n_ok,
                    enhance_sec, seg_sec, mes_sec, b2b_sec, total_sec,
                    enhance_provider, config_path, dll_dir, recipe_json, host_path,
                    started_at, finished_at, status, error_message
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    run_id,
                    wafer_key,
                    chip_key,
                    payload.get("date_folder", "") or "",
                    payload.get("lot_foup_id", "") or "",
                    payload.get("wafer_id", "") or "",
                    int(payload.get("chip_col") or 0),
                    int(payload.get("chip_row") or 0),
                    int(payload.get("fov_index") or 0),
                    payload.get("fov_folder", "") or "",
                    payload.get("input_path", "") or "",
                    payload.get("results_dir", "") or "",
                    payload.get("judgment", "") or "",
                    int(payload.get("final_bin") if payload.get("final_bin") is not None else 2),
                    int(payload.get("n_objects") or 0),
                    int(payload.get("n_ng") or 0),
                    int(payload.get("n_ok") or 0),
                    float(payload.get("enhance_sec") or 0),
                    float(payload.get("seg_sec") or 0),
                    float(payload.get("mes_sec") or 0),
                    float(payload.get("b2b_sec") or 0),
                    float(payload.get("total_sec") or 0),
                    payload.get("enhance_provider", "") or "",
                    payload.get("config_path", "") or "",
                    payload.get("dll_dir", "") or "",
                    recipe_json or "",
                    payload.get("host_path", "") or payload.get("input_path", "") or "",
                    started,
                    finished,
                    payload.get("status", "ok") or "ok",
                    payload.get("error_message", "") or "",
                ),
            )

            # Replace children for this run_id
            con.execute("DELETE FROM mes_objects WHERE run_id=?", (run_id,))
            con.execute("DELETE FROM artifacts WHERE run_id=?", (run_id,))
            con.execute("DELETE FROM timeline WHERE run_id=?", (run_id,))

            for s in payload.get("mes_objects") or []:
                if not isinstance(s, dict):
                    continue
                ratio = s.get("ratio", 0.0)
                try:
                    ratio_f = float(ratio) if ratio != float("inf") else 1e9
                except (TypeError, ValueError):
                    ratio_f = 0.0
                jud = s.get("judgment")
                if jud in (1, 8):
                    jud_s = "NG" if jud == 1 else "OK"
                elif isinstance(jud, str) and jud:
                    jud_s = jud
                else:
                    jud_s = "NG" if ratio_f >= 0.05 else "OK"
                con.execute(
                    """
                    INSERT INTO mes_objects(
                        run_id, row_id, layer_name, grid_row, grid_col, soh,
                        c1_volume, c2_volume, ratio, judgment, pitch_x, pitch_y,
                        z_min, z_max, y_min, y_max, x_min, x_max,
                        centroid_z, centroid_y, centroid_x, extra_json
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        run_id,
                        int(s.get("row_id") or 0),
                        str(s.get("layer_name") or ""),
                        int(s.get("grid_row") if s.get("grid_row") is not None else -1),
                        int(s.get("grid_col") if s.get("grid_col") is not None else -1),
                        float(s.get("soh") or 0),
                        float(s.get("c1_volume") or 0),
                        float(s.get("c2_volume") or 0),
                        ratio_f,
                        jud_s,
                        float(s.get("pitch_x") or 0),
                        float(s.get("pitch_y") or 0),
                        int(s.get("z_min") or 0),
                        int(s.get("z_max") or 0),
                        int(s.get("y_min") or 0),
                        int(s.get("y_max") or 0),
                        int(s.get("x_min") or 0),
                        int(s.get("x_max") or 0),
                        float(s.get("centroid_z") or 0),
                        float(s.get("centroid_y") or 0),
                        float(s.get("centroid_x") or 0),
                        "",
                    ),
                )

            for a in payload.get("artifacts") or []:
                if not isinstance(a, dict):
                    continue
                path = a.get("path") or ""
                if not path:
                    continue
                con.execute(
                    """
                    INSERT INTO artifacts(run_id, kind, path, exists_flag)
                    VALUES (?,?,?,?)
                    """,
                    (
                        run_id,
                        str(a.get("kind") or "file"),
                        str(path),
                        1 if os.path.exists(path) else 0,
                    ),
                )

            for t in payload.get("timeline") or []:
                if not isinstance(t, dict):
                    continue
                con.execute(
                    """
                    INSERT INTO timeline(run_id, ts, step, message, duration_sec)
                    VALUES (?,?,?,?,?)
                    """,
                    (
                        run_id,
                        t.get("ts") or finished,
                        str(t.get("step") or ""),
                        str(t.get("message") or ""),
                        float(t.get("duration_sec") or 0),
                    ),
                )

        return run_id

    # ------------------------------------------------------------------
    # Read API
    # ------------------------------------------------------------------
    def list_lots(self) -> List[Dict[str, Any]]:
        with self._conn() as con:
            rows = con.execute(
                """
                SELECT date_folder, lot_foup_id,
                       COUNT(DISTINCT wafer_id) AS n_wafers,
                       COUNT(*) AS n_fov,
                       SUM(CASE WHEN judgment='OK' THEN 1 ELSE 0 END) AS n_ok,
                       SUM(CASE WHEN judgment='NG' THEN 1 ELSE 0 END) AS n_ng
                FROM fov_runs
                GROUP BY date_folder, lot_foup_id
                ORDER BY MAX(finished_at) DESC
                """
            ).fetchall()
        return [dict(r) for r in rows]

    def list_lot_foup_parts(self) -> List[Dict[str, Any]]:
        """``list_lots()`` rows with ``lot_id`` / ``foup_id`` split for UI filters."""
        from inno3d.core.lot_foup import split_lot_foup

        out: List[Dict[str, Any]] = []
        for row in self.list_lots():
            lot_id, foup_id = split_lot_foup(str(row.get("lot_foup_id") or ""))
            d = dict(row)
            d["lot_id"] = lot_id
            d["foup_id"] = foup_id
            out.append(d)
        return out

    def list_wafers(
        self, date_folder: str = "", lot_foup_id: str = ""
    ) -> List[Dict[str, Any]]:
        q = """
            SELECT wafer_key, date_folder, lot_foup_id, wafer_id,
                   COUNT(*) AS n_fov,
                   SUM(CASE WHEN judgment='OK' THEN 1 ELSE 0 END) AS n_ok,
                   SUM(CASE WHEN judgment='NG' THEN 1 ELSE 0 END) AS n_ng,
                   SUM(n_objects) AS n_objects,
                   MAX(finished_at) AS last_run
            FROM fov_runs
            WHERE 1=1
        """
        args: List[Any] = []
        if date_folder:
            q += " AND date_folder=?"
            args.append(date_folder)
        if lot_foup_id:
            q += " AND lot_foup_id=?"
            args.append(lot_foup_id)
        q += " GROUP BY wafer_key ORDER BY last_run DESC"
        with self._conn() as con:
            rows = con.execute(q, args).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            n = int(d.get("n_fov") or 0)
            nok = int(d.get("n_ok") or 0)
            d["yield_pct"] = (100.0 * nok / n) if n else 0.0
            out.append(d)
        return out

    def list_fov_runs(
        self,
        wafer_key: str = "",
        judgment: str = "",
        search: str = "",
        limit: int = 500,
    ) -> List[Dict[str, Any]]:
        q = "SELECT * FROM fov_runs WHERE 1=1"
        args: List[Any] = []
        if wafer_key:
            q += " AND wafer_key=?"
            args.append(wafer_key)
        if judgment and judgment.upper() in ("OK", "NG", "ERROR", "PENDING"):
            q += " AND UPPER(judgment)=?"
            args.append(judgment.upper())
        if search:
            q += " AND (host_path LIKE ? OR results_dir LIKE ? OR CAST(chip_col AS TEXT)||','||CAST(chip_row AS TEXT) LIKE ?)"
            s = f"%{search}%"
            args.extend([s, s, s])
        q += " ORDER BY finished_at DESC LIMIT ?"
        args.append(int(limit))
        with self._conn() as con:
            rows = con.execute(q, args).fetchall()
        return [dict(r) for r in rows]

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        with self._conn() as con:
            row = con.execute(
                "SELECT * FROM fov_runs WHERE run_id=?", (run_id,)
            ).fetchone()
            if not row:
                return None
            d = dict(row)
            d["mes_objects"] = [
                dict(r)
                for r in con.execute(
                    "SELECT * FROM mes_objects WHERE run_id=? ORDER BY row_id",
                    (run_id,),
                ).fetchall()
            ]
            d["artifacts"] = [
                dict(r)
                for r in con.execute(
                    "SELECT * FROM artifacts WHERE run_id=?", (run_id,)
                ).fetchall()
            ]
            d["timeline"] = [
                dict(r)
                for r in con.execute(
                    "SELECT * FROM timeline WHERE run_id=? ORDER BY id",
                    (run_id,),
                ).fetchall()
            ]
        return d

    def wafer_chip_bins(self, wafer_key: str) -> List[Dict[str, Any]]:
        """Chip FinalBin rollup for wafer map coloring.

        Same rule as ``ChipCell.recompute_counts``:
          · any FOV NG → bin 1 (NG)
          · all 9 FOVs OK → bin 8 (OK)
          · otherwise → bin 2 (Pending)

        NOTE: Do **not** use MAX(final_bin) — FOV OK=8 would hide FOV NG=1.
        """
        with self._conn() as con:
            rows = con.execute(
                """
                SELECT chip_col, chip_row,
                       COUNT(DISTINCT fov_index) AS n_fov,
                       SUM(CASE WHEN UPPER(judgment)='NG' OR final_bin=1 THEN 1 ELSE 0 END) AS n_ng,
                       SUM(CASE WHEN UPPER(judgment)='OK' OR final_bin=8 THEN 1 ELSE 0 END) AS n_ok
                FROM fov_runs
                WHERE wafer_key=?
                GROUP BY chip_col, chip_row
                """,
                (wafer_key,),
            ).fetchall()
        out: List[Dict[str, Any]] = []
        for r in rows:
            d = dict(r)
            n_ng = int(d.get("n_ng") or 0)
            n_ok = int(d.get("n_ok") or 0)
            if n_ng > 0:
                d["final_bin"] = 1  # BIN_NG
            elif n_ok >= 9:
                d["final_bin"] = 8  # BIN_OK — full 9-point pass
            else:
                d["final_bin"] = 2  # BIN_PENDING
            out.append(d)
        return out

    def fov_point_yield(self, wafer_key: str = "") -> List[Dict[str, Any]]:
        q = """
            SELECT fov_index,
                   COUNT(*) AS n,
                   SUM(CASE WHEN judgment='OK' THEN 1 ELSE 0 END) AS n_ok
            FROM fov_runs
            WHERE fov_index > 0
        """
        args: List[Any] = []
        if wafer_key:
            q += " AND wafer_key=?"
            args.append(wafer_key)
        q += " GROUP BY fov_index ORDER BY fov_index"
        with self._conn() as con:
            rows = con.execute(q, args).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            n = int(d["n"] or 0)
            d["yield_pct"] = (100.0 * int(d["n_ok"] or 0) / n) if n else 0.0
            out.append(d)
        return out

    def ratio_histogram(
        self, wafer_key: str = "", bins: int = 12
    ) -> List[Tuple[float, int]]:
        """Return list of (bin_center, count) for MES ratio * 100."""
        q = "SELECT ratio FROM mes_objects m JOIN fov_runs r ON r.run_id=m.run_id WHERE ratio < 1e8"
        args: List[Any] = []
        if wafer_key:
            q += " AND r.wafer_key=?"
            args.append(wafer_key)
        with self._conn() as con:
            rows = con.execute(q, args).fetchall()
        vals = [float(r[0]) * 100.0 for r in rows if r[0] is not None]
        if not vals:
            return [(i, 0) for i in range(bins)]
        lo, hi = 0.0, max(20.0, max(vals))
        width = (hi - lo) / bins if bins > 0 else 1.0
        counts = [0] * bins
        for v in vals:
            idx = int((v - lo) / width)
            if idx >= bins:
                idx = bins - 1
            if idx < 0:
                idx = 0
            counts[idx] += 1
        return [(lo + (i + 0.5) * width, counts[i]) for i in range(bins)]

    def global_kpis(self, wafer_key: str = "") -> Dict[str, Any]:
        q = (
            "SELECT COUNT(*) n,"
            " SUM(CASE WHEN judgment='OK' THEN 1 ELSE 0 END) nok,"
            " SUM(CASE WHEN judgment='NG' THEN 1 ELSE 0 END) nng,"
            " SUM(n_objects) nob,"
            " SUM(n_ng) nbumpng"
            " FROM fov_runs WHERE 1=1"
        )
        args: List[Any] = []
        if wafer_key:
            q += " AND wafer_key=?"
            args.append(wafer_key)
        with self._conn() as con:
            r = con.execute(q, args).fetchone()
        n = int(r["n"] or 0)
        nok = int(r["nok"] or 0)
        return {
            "n_fov": n,
            "n_ok": nok,
            "n_ng": int(r["nng"] or 0),
            "n_objects": int(r["nob"] or 0),
            "n_bump_ng": int(r["nbumpng"] or 0),
            "yield_pct": (100.0 * nok / n) if n else 0.0,
        }

    # ------------------------------------------------------------------
    # SOH Analysis query API (3D Analysis tab)
    # ------------------------------------------------------------------
    def list_chips_for_wafer(self, wafer_key: str) -> List[Dict[str, Any]]:
        """Chip rollup with FOV counts + judgment for scope tree."""
        if not wafer_key:
            return []
        with self._conn() as con:
            rows = con.execute(
                """
                SELECT chip_col, chip_row, chip_key,
                       COUNT(*) AS n_fov,
                       SUM(CASE WHEN UPPER(judgment)='NG' OR final_bin=1 THEN 1 ELSE 0 END) AS n_ng,
                       SUM(CASE WHEN UPPER(judgment)='OK' OR final_bin=8 THEN 1 ELSE 0 END) AS n_ok,
                       SUM(n_objects) AS n_objects,
                       SUM(n_ng) AS n_obj_ng
                FROM fov_runs
                WHERE wafer_key=?
                GROUP BY chip_col, chip_row, chip_key
                ORDER BY chip_row, chip_col
                """,
                (wafer_key,),
            ).fetchall()
        out: List[Dict[str, Any]] = []
        for r in rows:
            d = dict(r)
            n_ng = int(d.get("n_ng") or 0)
            n_ok = int(d.get("n_ok") or 0)
            if n_ng > 0:
                d["judgment"] = "NG"
            elif n_ok >= 9:
                d["judgment"] = "OK"
            elif n_ok > 0 and n_ng == 0:
                d["judgment"] = "OK" if int(d.get("n_fov") or 0) < 9 else "OK"
            else:
                d["judgment"] = "PENDING"
            out.append(d)
        return out

    def list_mes_joined(
        self,
        *,
        wafer_key: str = "",
        run_ids: Optional[Iterable[str]] = None,
        chip_col: Optional[int] = None,
        chip_row: Optional[int] = None,
        fov_index: Optional[int] = None,
        layer_name: str = "",
        limit: int = 200000,
    ) -> List[Dict[str, Any]]:
        """MES objects joined with FOV run meta (for SOH analytics)."""
        q = """
            SELECT m.*,
                   r.wafer_key, r.chip_key, r.date_folder, r.lot_foup_id,
                   r.wafer_id, r.chip_col, r.chip_row,
                   r.fov_index, r.fov_folder, r.results_dir, r.input_path,
                   r.judgment AS run_judgment, r.final_bin, r.finished_at
            FROM mes_objects m
            JOIN fov_runs r ON r.run_id = m.run_id
            WHERE 1=1
        """
        args: List[Any] = []
        if wafer_key:
            q += " AND r.wafer_key=?"
            args.append(wafer_key)
        if run_ids is not None:
            ids = [str(x) for x in run_ids if x]
            if not ids:
                return []
            placeholders = ",".join("?" * len(ids))
            q += f" AND m.run_id IN ({placeholders})"
            args.extend(ids)
        if chip_col is not None:
            q += " AND r.chip_col=?"
            args.append(int(chip_col))
        if chip_row is not None:
            q += " AND r.chip_row=?"
            args.append(int(chip_row))
        if fov_index is not None:
            q += " AND r.fov_index=?"
            args.append(int(fov_index))
        if layer_name:
            q += " AND m.layer_name=?"
            args.append(layer_name)
        q += " ORDER BY r.chip_row, r.chip_col, r.fov_index, m.layer_name, m.row_id LIMIT ?"
        args.append(int(limit))
        with self._conn() as con:
            rows = con.execute(q, args).fetchall()
        return [dict(r) for r in rows]

    def aggregate_soh_by_layer(
        self,
        *,
        wafer_key: str = "",
        run_ids: Optional[Iterable[str]] = None,
        ng_threshold: float = 0.05,
    ) -> List[Dict[str, Any]]:
        """SQL aggregate SOH / void / NG per layer_name."""
        q = """
            SELECT m.layer_name AS key,
                   COUNT(*) AS n,
                   AVG(m.soh) AS mean_soh,
                   AVG(m.ratio) AS mean_ratio,
                   MAX(m.ratio) AS max_ratio,
                   AVG(m.c1_volume) AS mean_bump_vol,
                   SUM(m.c1_volume) AS total_bump_vol,
                   SUM(m.c2_volume) AS total_void_vol,
                   MIN(m.soh) AS min_soh,
                   MAX(m.soh) AS max_soh,
                   SUM(CASE WHEN m.ratio >= ? OR UPPER(COALESCE(m.judgment,''))='NG'
                            THEN 1 ELSE 0 END) AS n_ng
            FROM mes_objects m
            JOIN fov_runs r ON r.run_id = m.run_id
            WHERE m.soh IS NOT NULL
        """
        args: List[Any] = [float(ng_threshold)]
        if wafer_key:
            q += " AND r.wafer_key=?"
            args.append(wafer_key)
        if run_ids is not None:
            ids = [str(x) for x in run_ids if x]
            if not ids:
                return []
            placeholders = ",".join("?" * len(ids))
            q += f" AND m.run_id IN ({placeholders})"
            args.extend(ids)
        q += " GROUP BY m.layer_name"
        with self._conn() as con:
            rows = con.execute(q, args).fetchall()
        out: List[Dict[str, Any]] = []
        for r in rows:
            d = dict(r)
            n = int(d.get("n") or 0)
            n_ng = int(d.get("n_ng") or 0)
            d["ng_rate"] = (n_ng / n) if n else 0.0
            # sample std via second query would be heavy; fill later in Python if needed
            d["std_soh"] = 0.0
            out.append(d)
        # Compute std_soh in one pass for scoped rows (lightweight for demo sizes)
        std_map = self._soh_std_by_key(
            group_expr="m.layer_name",
            wafer_key=wafer_key,
            run_ids=run_ids,
        )
        for d in out:
            d["std_soh"] = float(std_map.get(str(d.get("key") or ""), 0.0))
        return out

    def aggregate_soh_by_fov(
        self,
        *,
        wafer_key: str = "",
        run_ids: Optional[Iterable[str]] = None,
        ng_threshold: float = 0.05,
    ) -> List[Dict[str, Any]]:
        """SQL aggregate SOH per FOV index (P1–P9)."""
        q = """
            SELECT r.fov_index AS key,
                   COUNT(*) AS n,
                   AVG(m.soh) AS mean_soh,
                   AVG(m.ratio) AS mean_ratio,
                   MAX(m.ratio) AS max_ratio,
                   MIN(m.soh) AS min_soh,
                   MAX(m.soh) AS max_soh,
                   SUM(CASE WHEN m.ratio >= ? OR UPPER(COALESCE(m.judgment,''))='NG'
                            THEN 1 ELSE 0 END) AS n_ng,
                   SUM(CASE WHEN UPPER(r.judgment)='NG' THEN 1 ELSE 0 END) AS n_run_ng,
                   COUNT(DISTINCT r.run_id) AS n_runs
            FROM mes_objects m
            JOIN fov_runs r ON r.run_id = m.run_id
            WHERE m.soh IS NOT NULL AND r.fov_index > 0
        """
        args: List[Any] = [float(ng_threshold)]
        if wafer_key:
            q += " AND r.wafer_key=?"
            args.append(wafer_key)
        if run_ids is not None:
            ids = [str(x) for x in run_ids if x]
            if not ids:
                return []
            placeholders = ",".join("?" * len(ids))
            q += f" AND m.run_id IN ({placeholders})"
            args.extend(ids)
        q += " GROUP BY r.fov_index ORDER BY r.fov_index"
        with self._conn() as con:
            rows = con.execute(q, args).fetchall()
        out: List[Dict[str, Any]] = []
        for r in rows:
            d = dict(r)
            n = int(d.get("n") or 0)
            n_ng = int(d.get("n_ng") or 0)
            d["ng_rate"] = (n_ng / n) if n else 0.0
            d["std_soh"] = 0.0
            out.append(d)
        std_map = self._soh_std_by_key(
            group_expr="CAST(r.fov_index AS TEXT)",
            wafer_key=wafer_key,
            run_ids=run_ids,
        )
        for d in out:
            d["std_soh"] = float(std_map.get(str(d.get("key") or ""), 0.0))
        return out

    def aggregate_soh_by_chip(
        self,
        *,
        wafer_key: str = "",
        run_ids: Optional[Iterable[str]] = None,
        ng_threshold: float = 0.05,
    ) -> List[Dict[str, Any]]:
        """SQL aggregate SOH per chip (col,row)."""
        q = """
            SELECT r.chip_col, r.chip_row,
                   printf('%d,%d', r.chip_col, r.chip_row) AS key,
                   COUNT(*) AS n,
                   AVG(m.soh) AS mean_soh,
                   AVG(m.ratio) AS mean_ratio,
                   MAX(m.ratio) AS max_ratio,
                   MIN(m.soh) AS min_soh,
                   MAX(m.soh) AS max_soh,
                   SUM(CASE WHEN m.ratio >= ? OR UPPER(COALESCE(m.judgment,''))='NG'
                            THEN 1 ELSE 0 END) AS n_ng,
                   COUNT(DISTINCT r.run_id) AS n_runs
            FROM mes_objects m
            JOIN fov_runs r ON r.run_id = m.run_id
            WHERE m.soh IS NOT NULL
        """
        args: List[Any] = [float(ng_threshold)]
        if wafer_key:
            q += " AND r.wafer_key=?"
            args.append(wafer_key)
        if run_ids is not None:
            ids = [str(x) for x in run_ids if x]
            if not ids:
                return []
            placeholders = ",".join("?" * len(ids))
            q += f" AND m.run_id IN ({placeholders})"
            args.extend(ids)
        q += " GROUP BY r.chip_col, r.chip_row ORDER BY r.chip_row, r.chip_col"
        with self._conn() as con:
            rows = con.execute(q, args).fetchall()
        out: List[Dict[str, Any]] = []
        for r in rows:
            d = dict(r)
            n = int(d.get("n") or 0)
            n_ng = int(d.get("n_ng") or 0)
            d["ng_rate"] = (n_ng / n) if n else 0.0
            out.append(d)
        return out

    def soh_kpis(
        self,
        *,
        wafer_key: str = "",
        run_ids: Optional[Iterable[str]] = None,
        ng_threshold: float = 0.05,
    ) -> Dict[str, Any]:
        """Combined FOV + MES KPIs for analysis scope."""
        run_q = "SELECT COUNT(*) n, SUM(CASE WHEN judgment='OK' THEN 1 ELSE 0 END) nok, SUM(CASE WHEN judgment='NG' THEN 1 ELSE 0 END) nng FROM fov_runs WHERE 1=1"
        mes_q = """
            SELECT COUNT(*) n, AVG(m.soh) mean_soh, AVG(m.ratio) mean_ratio,
                   SUM(CASE WHEN m.ratio >= ? OR UPPER(COALESCE(m.judgment,''))='NG'
                            THEN 1 ELSE 0 END) n_ng
            FROM mes_objects m JOIN fov_runs r ON r.run_id=m.run_id WHERE 1=1
        """
        run_args: List[Any] = []
        mes_args: List[Any] = [float(ng_threshold)]
        if wafer_key:
            run_q += " AND wafer_key=?"
            mes_q += " AND r.wafer_key=?"
            run_args.append(wafer_key)
            mes_args.append(wafer_key)
        if run_ids is not None:
            ids = [str(x) for x in run_ids if x]
            if not ids:
                return {
                    "n_fov": 0, "n_ok": 0, "n_ng": 0, "yield_pct": 0.0,
                    "n_objects": 0, "mean_soh": 0.0, "mean_ratio": 0.0,
                    "obj_ng_rate": 0.0,
                }
            ph = ",".join("?" * len(ids))
            run_q += f" AND run_id IN ({ph})"
            mes_q += f" AND m.run_id IN ({ph})"
            run_args.extend(ids)
            mes_args.extend(ids)
        with self._conn() as con:
            rr = con.execute(run_q, run_args).fetchone()
            mr = con.execute(mes_q, mes_args).fetchone()
        n = int(rr["n"] or 0)
        nok = int(rr["nok"] or 0)
        n_obj = int(mr["n"] or 0)
        n_obj_ng = int(mr["n_ng"] or 0)
        return {
            "n_fov": n,
            "n_ok": nok,
            "n_ng": int(rr["nng"] or 0),
            "yield_pct": (100.0 * nok / n) if n else 0.0,
            "n_objects": n_obj,
            "mean_soh": float(mr["mean_soh"] or 0.0),
            "mean_ratio": float(mr["mean_ratio"] or 0.0),
            "obj_ng_rate": (n_obj_ng / n_obj) if n_obj else 0.0,
        }

    def _soh_std_by_key(
        self,
        *,
        group_expr: str,
        wafer_key: str = "",
        run_ids: Optional[Iterable[str]] = None,
    ) -> Dict[str, float]:
        """Population std of SOH grouped by SQL expression (string key)."""
        q = f"""
            SELECT {group_expr} AS gkey, m.soh
            FROM mes_objects m
            JOIN fov_runs r ON r.run_id = m.run_id
            WHERE m.soh IS NOT NULL
        """
        args: List[Any] = []
        if wafer_key:
            q += " AND r.wafer_key=?"
            args.append(wafer_key)
        if run_ids is not None:
            ids = [str(x) for x in run_ids if x]
            if not ids:
                return {}
            placeholders = ",".join("?" * len(ids))
            q += f" AND m.run_id IN ({placeholders})"
            args.extend(ids)
        with self._conn() as con:
            rows = con.execute(q, args).fetchall()
        buckets: Dict[str, List[float]] = {}
        for r in rows:
            k = str(r["gkey"] if r["gkey"] is not None else "")
            try:
                buckets.setdefault(k, []).append(float(r["soh"]))
            except (TypeError, ValueError):
                continue
        return {k: _std(vals) for k, vals in buckets.items()}

    def wipe_all(self) -> None:
        """Delete all catalog rows; keep schema (v2, no RecipeID)."""
        with self._conn() as con:
            con.execute("PRAGMA foreign_keys=OFF")
            try:
                for table in (
                    "timeline",
                    "artifacts",
                    "mes_objects",
                    "fov_runs",
                    "chips",
                    "wafers",
                ):
                    con.execute(f"DELETE FROM {table}")
            finally:
                con.execute("PRAGMA foreign_keys=ON")

    def rebuild_clean_demo(self) -> int:
        """Wipe catalog and seed demo rows under new path layout. Returns # inserted."""
        self.wipe_all()
        return self.seed_demo_if_empty(force=True)

    def seed_demo_if_empty(self, force: bool = False) -> int:
        """Insert synthetic FOV runs for UI demo when DB empty.

        Paths follow mass-production layout (no RecipeID)::

            {date}/{LotID_FoupID}/{WaferID}/{Chip}/{FOV}/Input.tiff

        Returns number of FOV runs inserted.
        """
        with self._conn() as con:
            n = con.execute("SELECT COUNT(*) FROM fov_runs").fetchone()[0]
        if n and not force:
            return 0
        if n and force:
            self.wipe_all()

        date_folder = "26_07_18"
        lot_foup = "LOT240701_F01"
        inserted = 0
        for wi, wid in enumerate(("WAFER05", "WAFER06", "WAFER07")):
            for ci in range(3):
                for fi in range(1, 4):
                    jud = "NG" if (ci + fi + wi) % 7 == 0 else "OK"
                    n_obj = 40 + (ci * 3) + fi
                    n_ng = max(1, n_obj // 12) if jud == "NG" else max(0, n_obj // 40)
                    chip_col = 10 + ci
                    chip_row = 12 + wi
                    chip_folder = f"Chip_{chip_col}_{chip_row}"
                    fov_folder = f"FOV_P{fi}"
                    # Canonical host path (no RecipeID segment)
                    base = (
                        f"E:/demo/{date_folder}/{lot_foup}/{wid}/"
                        f"{chip_folder}/{fov_folder}"
                    )
                    input_path = f"{base}/Input.tiff"
                    results_dir = f"{base}/Results"
                    objs = []
                    for k in range(min(8, n_obj)):
                        ratio = 0.12 if (jud == "NG" and k < 2) else 0.02 + 0.005 * k
                        objs.append(
                            {
                                "row_id": k + 1,
                                "layer_name": f"Layer {(k % 4) + 1}",
                                "grid_row": k // 4,
                                "grid_col": k % 4,
                                "soh": 11.5 + k * 0.1,
                                "c1_volume": 800 + k * 10,
                                "c2_volume": 800 * ratio,
                                "ratio": ratio,
                                "judgment": "NG" if ratio >= 0.05 else "OK",
                                "pitch_x": 0.3,
                                "pitch_y": 0.28,
                                "z_min": 20,
                                "z_max": 40,
                                "y_min": 10,
                                "y_max": 50,
                                "x_min": 10,
                                "x_max": 50,
                                "centroid_z": 30,
                                "centroid_y": 30,
                                "centroid_x": 30,
                            }
                        )
                    self.record_fov_run(
                        {
                            "date_folder": date_folder,
                            "lot_foup_id": lot_foup,
                            "lot_id": "LOT240701",
                            "foup_id": "F01",
                            "wafer_id": wid,
                            "chip_col": chip_col,
                            "chip_row": chip_row,
                            "chip_folder": chip_folder,
                            "fov_index": fi,
                            "fov_folder": fov_folder,
                            "input_path": input_path,
                            "results_dir": results_dir,
                            "host_path": input_path,
                            "judgment": jud,
                            "final_bin": 1 if jud == "NG" else 8,
                            "n_objects": n_obj,
                            "n_ng": n_ng,
                            "n_ok": n_obj - n_ng,
                            "enhance_sec": 40 + fi,
                            "seg_sec": 30 + ci,
                            "mes_sec": 2.5,
                            "b2b_sec": 1.2,
                            "total_sec": 75 + fi,
                            "enhance_provider": "cuda",
                            "config_path": "config_HBM_c2848_M.txt",
                            "status": "ok",
                            "mes_objects": objs,
                            "timeline": [
                                {"ts": _utc_now(), "step": "recv", "message": "TCP receive Input.tiff"},
                                {
                                    "ts": _utc_now(),
                                    "step": "enhance",
                                    "message": "Enhance 900 slices · provider=cuda",
                                    "duration_sec": 40 + fi,
                                },
                                {
                                    "ts": _utc_now(),
                                    "step": "seg",
                                    "message": "SEG 12 layers",
                                    "duration_sec": 30 + ci,
                                },
                                {
                                    "ts": _utc_now(),
                                    "step": "mes",
                                    "message": f"MES · {n_obj} objects · NG {n_ng}",
                                    "duration_sec": 2.5,
                                },
                                {
                                    "ts": _utc_now(),
                                    "step": "b2b",
                                    "message": f"B2B · FinalBin={jud}",
                                    "duration_sec": 1.2,
                                },
                            ],
                            "artifacts": [
                                {
                                    "kind": "csv_mes",
                                    "path": f"{results_dir}/object_statistics.csv",
                                }
                            ],
                        }
                    )
                    inserted += 1
        return inserted


def get_db(db_path: Optional[os.PathLike] = None) -> InspectionDB:
    global _default_db
    if db_path is not None:
        return InspectionDB(db_path)
    if _default_db is None:
        _default_db = InspectionDB()
    return _default_db


def record_online_fov_completion(payload: Dict[str, Any]) -> str:
    """Convenience for Online pipeline."""
    return get_db().record_fov_run(payload)
