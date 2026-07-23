# inno3d/app/bootstrap.py
# -----------------------------------------------------------------------
# Application bootstrap — entry point wrapper (Phase 6)
#
# Per REFACTOR_PLAN_PROFESSIONAL.md Phase 6 target:
#   from inno3d.app.bootstrap import run
#   if __name__ == "__main__":
#       run()
#
# NOTE: main.py remains the actual entry point for PyInstaller builds
# and direct runs. bootstrap.py provides the clean API for future use.
# -----------------------------------------------------------------------
"""Bootstrap entry for inno3d application."""
import sys


def run():
    """Launch the Inno3D application."""
    import runpy
    from pathlib import Path
    main_path = str(Path(__file__).resolve().parents[2] / "main.py")
    runpy.run_path(main_path, run_name="__main__")
