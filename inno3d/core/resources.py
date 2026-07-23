"""Backward-compatible shim — canonical location is now ``inno3d.infra.paths``.

All public symbols are re-exported so existing ``from inno3d.core.resources import …``
statements continue to work without modification.
"""
from inno3d.infra.paths import (        # noqa: F401 — re-exports
    project_root,
    app_runtime_dir,
    resource_path,
    default_dll_dir,
    default_config_path,
    config_dir,
    list_config_files,
    RECIPE_EXTENSIONS,
    log_dir,
    data_dir,
    app_config_ini_path,
)
