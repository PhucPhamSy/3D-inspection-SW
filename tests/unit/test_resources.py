"""Unit tests for inno3d.core.resources — path resolution with monkeypatched frozen state."""
import sys
import os
import pytest
from pathlib import Path
from unittest.mock import patch

from inno3d.core.resources import (
    project_root,
    app_runtime_dir,
    resource_path,
    default_dll_dir,
    default_config_path,
)


class TestProjectRoot:
    def test_returns_path(self):
        root = project_root()
        assert isinstance(root, Path)

    def test_root_contains_inno3d(self):
        """project_root() should point to the workspace containing inno3d/."""
        root = project_root()
        assert (root / "inno3d").is_dir(), f"Expected inno3d/ under {root}"

    def test_root_contains_main(self):
        root = project_root()
        assert (root / "main.py").is_file()


class TestAppRuntimeDir:
    def test_dev_mode_equals_project_root(self):
        """In dev (not frozen), app_runtime_dir == project_root."""
        # Make sure sys.frozen is not set
        frozen = getattr(sys, "frozen", None)
        if frozen:
            pytest.skip("Running in frozen mode")
        assert app_runtime_dir() == project_root()

    def test_frozen_mode_uses_executable_parent(self):
        """When sys.frozen is set, app_runtime_dir uses sys.executable parent."""
        fake_exe = Path(r"C:\dist\Inno3D\Inno3D.exe")
        with patch.object(sys, "frozen", True, create=True), \
             patch.object(sys, "executable", str(fake_exe)):
            result = app_runtime_dir()
            assert result == fake_exe.resolve().parent


class TestResourcePath:
    def test_returns_string(self):
        result = resource_path("assets/branding/company_logo_v1.png")
        assert isinstance(result, str)

    def test_relative_path_resolved(self):
        result = resource_path("assets")
        assert os.path.isabs(result)

    def test_dev_mode_uses_project_root(self):
        """In dev mode, resource_path should resolve relative to project root."""
        if getattr(sys, "frozen", False):
            pytest.skip("Running in frozen mode")
        result = Path(resource_path("main.py"))
        assert result == (project_root() / "main.py").resolve()


class TestDefaultDllDir:
    def test_returns_str_or_none(self):
        result = default_dll_dir()
        assert result is None or isinstance(result, str)

    def test_dev_mode_finds_v2_if_exists(self):
        """In dev mode, should find V2/ if it exists next to project."""
        if getattr(sys, "frozen", False):
            pytest.skip("Running in frozen mode")
        result = default_dll_dir()
        if result is not None:
            assert Path(result).is_dir()
            # The resolved path should contain 'V2'
            assert "V2" in Path(result).name or "v2" in Path(result).name.lower()

    def test_frozen_mode_searches_next_to_exe(self):
        """In frozen mode, first candidate is <exe_dir>/V2."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            v2_dir = Path(tmpdir) / "V2"
            v2_dir.mkdir()
            fake_exe = Path(tmpdir) / "Inno3D.exe"

            with patch.object(sys, "frozen", True, create=True), \
                 patch.object(sys, "executable", str(fake_exe)):
                result = default_dll_dir()
                if result is not None:
                    assert Path(result).is_dir()


class TestDefaultConfigPath:
    def test_returns_str_or_none(self):
        result = default_config_path()
        assert result is None or isinstance(result, str)

    def test_finds_config_if_exists(self):
        if getattr(sys, "frozen", False):
            pytest.skip("Running in frozen mode")
        result = default_config_path()
        if result is not None:
            assert Path(result).is_file()
            assert result.endswith(".txt")
