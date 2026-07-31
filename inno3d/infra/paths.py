"""Portable path resolution for Inno3D Inspection.

Expanded from ``inno3d.core.resources`` — canonical location for all
filesystem-path helpers used by native loaders, config, logging, and
packaging.

Layer: infra (may import stdlib only; no Qt, no VTK).
"""
import os
from pathlib import Path
import sys
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


def default_dll_dir() -> str | None:
    """Resolve native DLL folder for SEG / MES / B2B / ENH.

    Search order:
      0. ``app_config.ini`` ``[DLL] dir`` (Teaching browse choice)
      1. <exe_or_project>/V2
      2. Parent-of-project shared all_v2/V2 (dev only)
      3. Parent-of-project shared all_v2/V3 (dev only)
    """
    try:
        from inno3d.app.settings import get_dll_dir as settings_dll_dir

        override = settings_dll_dir()
        if override:
            return override
    except Exception:
        pass

    candidates = [
        app_runtime_dir() / "V2",
    ]
    if not getattr(sys, "frozen", False):
        # Dev monorepo layout: …/all_v2/HBM_frontend_backend_v12 and …/all_v2/V2|V3
        parent = project_root().parent
        candidates.append(parent / "V2")
        candidates.append(parent / "V3")

    for c in candidates:
        try:
            if c.is_dir():
                return str(c.resolve())
        except OSError:
            continue
    return None


# Recipe files are parameter-only (no INPUT_PATH / OUTPUT_DIR).
# Plain KEY = value text — fast line parse, human-editable, DLL-native.
RECIPE_EXTENSIONS = (".txt",)


def config_dir() -> Path:
    """Directory of inspection recipes shipped next to the app.

    Prefer ``<exe_or_project>/config`` (populated by build). Dev fallback:
    project ``config/``. Does not create the folder.
    """
    candidates = [app_runtime_dir() / "config"]
    if not getattr(sys, "frozen", False):
        candidates.append(project_root() / "config")
        try:
            candidates.append(Path(resource_path("config")))
        except Exception:
            pass
    for d in candidates:
        try:
            if d.is_dir():
                return d.resolve()
        except OSError:
            continue
    return (app_runtime_dir() / "config")


def list_config_files() -> list[Path]:
    """Sorted list of recipe files under :func:`config_dir` (basename order)."""
    d = config_dir()
    if not d.is_dir():
        return []
    files: list[Path] = []
    try:
        for p in d.iterdir():
            if p.is_file() and p.suffix.lower() in RECIPE_EXTENSIONS:
                files.append(p.resolve())
    except OSError:
        return []
    return sorted(files, key=lambda p: p.name.lower())


def default_config_path() -> str | None:
    """Resolve a default HBM recipe for first launch (prefer known names)."""
    preferred = (
        "config_HBM_c2848_M.txt",
        "config_HBM.txt",
    )
    by_name = {p.name.lower(): p for p in list_config_files()}
    for name in preferred:
        hit = by_name.get(name.lower())
        if hit is not None:
            return str(hit)
    # Any remaining recipe in config/
    files = list_config_files()
    if files:
        return str(files[0])
    # Last-resort search (legacy absolute / monorepo layouts in dev only)
    if not getattr(sys, "frozen", False):
        for d in (project_root().parent, Path(r"E:\semiconductor\HBM_DEV_FOR_PROD\DEV\all_v2")):
            for name in preferred:
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
