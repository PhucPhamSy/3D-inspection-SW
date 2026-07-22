# conftest.py — project-root pytest configuration
#
# Adds the repo root to sys.path so that `import inno3d` works
# regardless of how pytest is invoked (e.g. via `conda run -n inno3d_ai pytest`).
import sys
import pathlib

# Ensure the project root (this file's directory) is first on sys.path.
_ROOT = pathlib.Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
