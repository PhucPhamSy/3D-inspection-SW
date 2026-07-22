"""Portable path resolution for Inno3D Inspection.

Expanded from ``inno3d.core.resources`` — canonical location for all
filesystem-path helpers used by native loaders, config, logging, and
packaging.

Layer: infra (may import stdlib only; no Qt, no VTK).
"""
import os
import sys
from pathlib import Path
from typing import Optional


def project_root() -> Path:
    """Source-tree root (…/HBM_frontend_backend_v12). Not valid for frozen runtime data."""
    return Path(__file__).resolve().parents[2]


def app_runtime_dir() -> Path:
    """Directory next to the running app (exe folder when frozen, project root in dev).

    Native DLLs (V2/), app_config.ini, and Inno3D_Data live here — not inside
    PyInstaller's temporary _MEIPASS extract folder.
    """
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return project_root()


def resource_path(relative_path: str) -> str:
    """Get absolute path to a bundled resource for dev and PyInstaller."""
    try:
        base_path = Path(sys._MEIPASS)
    except Exception:
        base_path = project_root()

    return str((base_path / relative_path).resolve())


def default_dll_dir() -> Optional[str]:
    """Resolve portable V2/ native DLL folder for SEG / MES / B2B / ENH.

    Search order:
      1. <exe_or_project>/V2
      2. Parent-of-project shared all_v2/V2 (dev only)
      3. Legacy absolute path used on the main dev machine (dev only)
    """
    candidates = [
        app_runtime_dir() / "V2",
    ]
    if not getattr(sys, "frozen", False):
        # Dev monorepo layout: …/all_v2/HBM_frontend_backend_v12 and …/all_v2/V2
        candidates.append(project_root().parent / "V2")

    for c in candidates:
        try:
            if c.is_dir():
                return str(c.resolve())
        except OSError:
            continue
    return None


def default_config_path() -> Optional[str]:
    """Resolve a default HBM config file for first launch."""
    names = (
        "config_HBM_c2848_M.txt",
        "config_HBM.txt",
    )
    search_dirs = [
        app_runtime_dir() / "config",
        project_root() / "config" if not getattr(sys, "frozen", False) else None,
        Path(resource_path("config")),
    ]
    if not getattr(sys, "frozen", False):
        search_dirs.append(project_root().parent)
        search_dirs.append(Path(r"E:\semiconductor\HBM_DEV_FOR_PROD\DEV\all_v2"))

    for d in search_dirs:
        if d is None:
            continue
        for name in names:
            p = Path(d) / name
            try:
                if p.is_file():
                    return str(p.resolve())
            except OSError:
                continue
    return None


def log_dir() -> Path:
    """Logging output directory: Inno3D_Logs/ next to the running app."""
    d = app_runtime_dir() / "Inno3D_Logs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def data_dir() -> Path:
    """User/inspection data directory: Inno3D_Data/ next to the running app."""
    d = app_runtime_dir() / "Inno3D_Data"
    d.mkdir(parents=True, exist_ok=True)
    return d


def app_config_ini_path() -> Path:
    """Resolve app_config.ini location for dev and packaged builds."""
    return app_runtime_dir() / "app_config.ini"
