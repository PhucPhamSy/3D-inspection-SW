"""Lot / FOUP parse & display helpers (shared by Line Pulse + Analysis).

Storage key remains ``lot_foup_id`` (e.g. ``LOT240701_F01``).
These helpers only split for UI filters and breadcrumbs.
"""
from __future__ import annotations

import re
from typing import Tuple


def split_lot_foup(lot_foup_id: str) -> Tuple[str, str]:
    """Split ``LotID_FoupID`` on the *last* underscore.

    Examples
    --------
    >>> split_lot_foup("LOT240701_F01")
    ('LOT240701', 'F01')
    >>> split_lot_foup("LOT240701")
    ('LOT240701', '')
    >>> split_lot_foup("")
    ('', '')
    """
    name = (lot_foup_id or "").strip()
    if not name:
        return "", ""
    if "_" not in name:
        return name, ""
    lot, foup = name.rsplit("_", 1)
    return lot.strip(), foup.strip()


def format_lot_foup_display(lot_id: str, foup_id: str) -> str:
    """Human-readable Lot · FOUP line.

    >>> format_lot_foup_display("LOT240701", "F01")
    'LOT240701 · FOUP 01'
    >>> format_lot_foup_display("LOT240701", "01")
    'LOT240701 · FOUP 01'
    >>> format_lot_foup_display("LOT240701", "")
    'LOT240701'
    """
    lot = (lot_id or "").strip()
    foup = (foup_id or "").strip()
    if not lot and not foup:
        return "—"
    if not foup:
        return lot or "—"
    # Strip leading F for display number; keep digits-only when possible
    foup_num = foup
    if foup.upper().startswith("F") and len(foup) > 1:
        foup_num = foup[1:]
    return f"{lot} · FOUP {foup_num}" if lot else f"FOUP {foup_num}"


def parse_date_folder_display(date_folder: str) -> str:
    """Map folder ``YY_MM_DD`` / ``DD_MM_YY`` style to ``YYYY-MM-DD`` when parseable.

    >>> parse_date_folder_display("26_07_25")
    '2026-07-25'
    >>> parse_date_folder_display("2026-07-25")
    '2026-07-25'
    >>> parse_date_folder_display("raw_folder")
    'raw_folder'
    """
    raw = (date_folder or "").strip()
    if not raw:
        return "—"
    # Already ISO-ish
    m_iso = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", raw)
    if m_iso:
        return raw
    # YY_MM_DD (fab folder convention used in this product)
    m = re.fullmatch(r"(\d{2})_(\d{2})_(\d{2})", raw)
    if m:
        yy, mm, dd = m.group(1), m.group(2), m.group(3)
        return f"20{yy}-{mm}-{dd}"
    return raw


def join_lot_foup(lot_id: str, foup_id: str) -> str:
    """Rebuild storage ``lot_foup_id`` from split parts."""
    lot = (lot_id or "").strip()
    foup = (foup_id or "").strip()
    if lot and foup:
        return f"{lot}_{foup}"
    return lot or foup or ""
