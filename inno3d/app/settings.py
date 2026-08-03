"""Centralized application settings — reads ``app_config.ini`` next to exe.

Provides typed accessors with safe fallbacks. All ini reads go through
this module so config changes are auditable in one place.

Layer: app (may import infra; no Qt).
"""
import configparser
import os
from pathlib import Path
from typing import Optional, Tuple

from inno3d.infra.paths import app_config_ini_path

_TRUE_VALUES = {"1", "true", "yes", "on", "y"}


def _read_ini() -> configparser.ConfigParser:
    """Read app_config.ini once per call (cheap for single-file ini)."""
    cfg = configparser.ConfigParser()
    ini = app_config_ini_path()
    if ini.exists():
        try:
            cfg.read(str(ini), encoding="utf-8")
        except Exception:
            pass
    return cfg


# ──────────────────────────────────────
# [APPEARANCE]
# ──────────────────────────────────────
def get_theme() -> str:
    """Return normalized theme name ('dark' or 'light')."""
    env = os.environ.get("INNO3D_THEME", "").strip().lower()
    if env in ("dark", "light"):
        return env
    raw = _read_ini().get("APPEARANCE", "theme", fallback="dark")
    return raw.strip().lower() if raw.strip().lower() in ("dark", "light") else "dark"


def set_theme(theme_name: str) -> None:
    """Persist theme preference to app_config.ini."""
    mode = theme_name.strip().lower()
    if mode not in ("dark", "light"):
        mode = "dark"
    ini = app_config_ini_path()
    cfg = configparser.ConfigParser()
    if ini.exists():
        try:
            cfg.read(str(ini), encoding="utf-8")
        except Exception:
            pass
    if "APPEARANCE" not in cfg:
        cfg["APPEARANCE"] = {}
    cfg["APPEARANCE"]["theme"] = mode
    try:
        with open(ini, "w", encoding="utf-8") as f:
            cfg.write(f)
    except Exception as e:
        print(f"[SETTINGS] Failed to save theme: {e}")


# ──────────────────────────────────────
# [DLL]
# ──────────────────────────────────────
def get_dll_dir() -> str | None:
    """DLL override from ini; ``None`` → use ``default_dll_dir()`` fallback."""
    raw = _read_ini().get("DLL", "dir", fallback="").strip()
    if raw and Path(raw).is_dir():
        return str(Path(raw).resolve())
    return None


def set_dll_dir(dll_dir: str | None) -> None:
    """Persist Teaching/Online native DLL folder to app_config.ini [DLL] dir."""
    ini = app_config_ini_path()
    cfg = configparser.ConfigParser()
    if ini.exists():
        try:
            cfg.read(str(ini), encoding="utf-8")
        except Exception:
            pass
    if "DLL" not in cfg:
        cfg["DLL"] = {}
    if dll_dir and Path(dll_dir).is_dir():
        cfg["DLL"]["dir"] = str(Path(dll_dir).resolve())
    else:
        cfg["DLL"]["dir"] = ""
    try:
        ini.parent.mkdir(parents=True, exist_ok=True)
        with open(ini, "w", encoding="utf-8") as f:
            cfg.write(f)
    except Exception as e:
        print(f"[SETTINGS] Failed to save DLL dir: {e}")


# ──────────────────────────────────────
# [ONLINE]
# ──────────────────────────────────────
def get_online_port() -> int:
    """TCP port for Online mode (default 8000)."""
    try:
        return int(_read_ini().get("ONLINE", "port", fallback="8000"))
    except (ValueError, TypeError):
        return 8000


# ──────────────────────────────────────
# [VOLUME_3D]
# ──────────────────────────────────────
VOLUME_3D_BUDGET_MB_OPTIONS: Tuple[int, ...] = (512, 768, 1024, 1536, 2048)
DEFAULT_3D_UPLOAD_BUDGET_MB = 512


def _normalize_budget_mb(raw_mb, *, restrict_options: bool = True) -> int:
    try:
        mb = int(float(raw_mb))
    except (TypeError, ValueError):
        mb = DEFAULT_3D_UPLOAD_BUDGET_MB
    mb = max(16, mb)
    if not restrict_options or mb in VOLUME_3D_BUDGET_MB_OPTIONS:
        return mb
    return min(VOLUME_3D_BUDGET_MB_OPTIONS, key=lambda option: abs(option - mb))


def _write_ini_section(section: str, values: dict[str, str]) -> None:
    ini = app_config_ini_path()
    cfg = configparser.ConfigParser()
    if ini.exists():
        try:
            cfg.read(str(ini), encoding="utf-8")
        except Exception:
            pass
    if section not in cfg:
        cfg[section] = {}
    cfg[section].update(values)
    try:
        ini.parent.mkdir(parents=True, exist_ok=True)
        with open(ini, "w", encoding="utf-8") as f:
            cfg.write(f)
    except Exception as e:
        print(f"[SETTINGS] Failed to save {section}: {e}")


def get_3d_upload_budget_mb() -> int:
    """Return 3D upload budget in MiB (env overrides ini)."""
    env = os.environ.get("INNO3D_3D_UPLOAD_BUDGET_MB", "").strip()
    if env:
        return _normalize_budget_mb(env, restrict_options=False)
    raw = _read_ini().get(
        "VOLUME_3D",
        "upload_budget_mb",
        fallback=str(DEFAULT_3D_UPLOAD_BUDGET_MB),
    )
    return _normalize_budget_mb(raw)


def set_3d_upload_budget_mb(budget_mb: int) -> None:
    """Persist 3D upload budget to app_config.ini [VOLUME_3D]."""
    mb = _normalize_budget_mb(budget_mb)
    _write_ini_section("VOLUME_3D", {"upload_budget_mb": str(mb)})


def get_large_volume_engine_enabled() -> bool:
    """Return whether the large volume engine is enabled (env overrides ini)."""
    env = os.environ.get("INNO3D_LARGE_VOLUME_ENGINE", "").strip().lower()
    if env in _TRUE_VALUES:
        return True
    raw = _read_ini().get("VOLUME_3D", "large_volume_engine", fallback="false").strip().lower()
    return raw in _TRUE_VALUES


def set_large_volume_engine_enabled(enabled: bool) -> None:
    """Persist large volume engine flag to app_config.ini [VOLUME_3D]."""
    _write_ini_section(
        "VOLUME_3D",
        {"large_volume_engine": "true" if enabled else "false"},
    )
