import pathlib as _pathlib

# Single source of truth: repo-root VERSION file.
# Falls back to hardcoded string inside a frozen (PyInstaller) build.
try:
    _version_file = _pathlib.Path(__file__).resolve().parent.parent / "VERSION"
    __version__: str = _version_file.read_text(encoding="utf-8").strip()
except FileNotFoundError:
    __version__ = "1.2.0"  # fallback for frozen / editable installs
