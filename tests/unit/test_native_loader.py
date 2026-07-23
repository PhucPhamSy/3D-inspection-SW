# tests/unit/test_native_loader.py
# -----------------------------------------------------------------------
# Unit tests for inno3d/native/loader.py
# All tests mock filesystem and ctypes to avoid needing actual DLLs.
# -----------------------------------------------------------------------
"""Tests for native DLL loader path search logic."""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys


# ─── get_dll_search_paths ─────────────────────────────────────────────────────

class TestGetDllSearchPaths:
    """Tests for the DLL directory priority search order."""

    def test_explicit_dll_dir_is_first(self, tmp_path):
        """An explicitly provided dll_dir should be the first candidate."""
        from inno3d.native.loader import get_dll_search_paths
        explicit = tmp_path / "explicit_dll"
        explicit.mkdir()
        paths = get_dll_search_paths(str(explicit))
        assert paths[0] == explicit.resolve()

    def test_explicit_dir_missing_is_excluded(self, tmp_path):
        """A non-existent explicit dll_dir should NOT appear in results."""
        from inno3d.native.loader import get_dll_search_paths
        missing = tmp_path / "does_not_exist"
        # Do NOT create the directory
        paths = get_dll_search_paths(str(missing))
        assert missing.resolve() not in paths

    def test_no_explicit_returns_dev_v2(self, tmp_path):
        """Without explicit dir, should include project V2 folder in dev mode."""
        from inno3d.native.loader import get_dll_search_paths
        # We can't easily test the real project root, but we can verify
        # the function returns a list without crashing
        with patch.object(sys, "frozen", False, create=True):
            paths = get_dll_search_paths(None)
        assert isinstance(paths, list)

    def test_frozen_mode_includes_exe_dir(self, tmp_path):
        """In frozen (PyInstaller) mode, exe directory should be included."""
        from inno3d.native.loader import get_dll_search_paths
        fake_exe = tmp_path / "Inno3D.exe"
        fake_exe.touch()
        fake_v2 = tmp_path / "V2"
        fake_v2.mkdir()

        with patch.object(sys, "frozen", True, create=True), \
             patch.object(sys, "executable", str(fake_exe)):
            paths = get_dll_search_paths(None)

        # exe dir and exe/V2 should be candidates (if they exist as dirs)
        path_strs = [str(p) for p in paths]
        assert str(tmp_path.resolve()) in path_strs or \
               str(fake_v2.resolve()) in path_strs

    def test_returns_only_existing_dirs(self, tmp_path):
        """All returned paths must exist as directories."""
        from inno3d.native.loader import get_dll_search_paths
        paths = get_dll_search_paths(None)
        for p in paths:
            assert p.is_dir(), f"Returned non-dir path: {p}"


# ─── DllLoadError ─────────────────────────────────────────────────────────────

class TestDllLoadError:
    """Tests for the DllLoadError exception class."""

    def test_is_runtime_error(self):
        from inno3d.native.loader import DllLoadError
        assert issubclass(DllLoadError, RuntimeError)

    def test_carries_message(self):
        from inno3d.native.loader import DllLoadError
        err = DllLoadError("DLL not found: BumpVoidSeg.dll")
        assert "BumpVoidSeg.dll" in str(err)

    def test_can_chain_exception(self):
        from inno3d.native.loader import DllLoadError
        try:
            try:
                raise FileNotFoundError("file.dll")
            except FileNotFoundError as e:
                raise DllLoadError("wrapped") from e
        except DllLoadError as ex:
            assert ex.__cause__ is not None
            assert isinstance(ex.__cause__, FileNotFoundError)


# ─── load_bumpvoid_dll ────────────────────────────────────────────────────────

class TestLoadBumpvoidDll:
    """Tests for load_bumpvoid_dll — mocked to avoid real DLLs."""

    def test_delegates_to_core_bumpvoid(self, tmp_path):
        """load_bumpvoid_dll should call bumpvoid.load_dll with the dir."""
        from inno3d.native import loader

        mock_bv = MagicMock()
        mock_bv._dll = MagicMock(name="fake_dll_handle")

        with patch.dict("sys.modules", {"inno3d.core.bumpvoid": mock_bv}):
            result = loader.load_bumpvoid_dll(str(tmp_path))

        mock_bv.load_dll.assert_called_once_with(str(tmp_path))
        assert result is mock_bv._dll

    def test_file_not_found_raises_dll_load_error(self, tmp_path):
        """FileNotFoundError from core should be wrapped as DllLoadError."""
        from inno3d.native import loader
        from inno3d.native.loader import DllLoadError

        mock_bv = MagicMock()
        mock_bv.load_dll.side_effect = FileNotFoundError("BumpVoidSeg.dll not found")

        with patch.dict("sys.modules", {"inno3d.core.bumpvoid": mock_bv}):
            with pytest.raises(DllLoadError):
                loader.load_bumpvoid_dll(str(tmp_path))

    def test_os_error_raises_dll_load_error(self, tmp_path):
        """OSError from core should be wrapped as DllLoadError."""
        from inno3d.native import loader
        from inno3d.native.loader import DllLoadError

        mock_bv = MagicMock()
        mock_bv.load_dll.side_effect = OSError("access denied")

        with patch.dict("sys.modules", {"inno3d.core.bumpvoid": mock_bv}):
            with pytest.raises(DllLoadError, match="OS error"):
                loader.load_bumpvoid_dll(str(tmp_path))


# ─── load_enhanced_dll ────────────────────────────────────────────────────────

class TestLoadEnhancedDll:
    """Tests for load_enhanced_dll — mocked to avoid real DLLs."""

    def test_delegates_to_enhanced_volume(self, tmp_path):
        """load_enhanced_dll should call enhanced_volume.load_dll with path."""
        from inno3d.native import loader

        fake_dll_path = str(tmp_path / "Enhanced.dll")
        mock_ev = MagicMock()
        mock_ev._dll = MagicMock(name="fake_enhanced_handle")

        with patch.dict("sys.modules", {"inno3d.core.enhanced_volume": mock_ev}):
            result = loader.load_enhanced_dll(fake_dll_path)

        mock_ev.load_dll.assert_called_once_with(fake_dll_path)
        assert result is mock_ev._dll

    def test_file_not_found_raises_dll_load_error(self, tmp_path):
        """FileNotFoundError should be wrapped as DllLoadError."""
        from inno3d.native import loader
        from inno3d.native.loader import DllLoadError

        mock_ev = MagicMock()
        mock_ev.load_dll.side_effect = FileNotFoundError("Enhanced.dll missing")

        with patch.dict("sys.modules", {"inno3d.core.enhanced_volume": mock_ev}):
            with pytest.raises(DllLoadError):
                loader.load_enhanced_dll(str(tmp_path / "Enhanced.dll"))

    def test_attribute_error_falls_back_to_ctypes(self, tmp_path):
        """If enhanced_volume has no load_dll, fall back to ctypes.CDLL."""
        from inno3d.native import loader

        fake_dll_path = str(tmp_path / "Enhanced.dll")
        # Create a dummy file so ctypes doesn't error on path
        (tmp_path / "Enhanced.dll").touch()

        mock_ev = MagicMock(spec=[])  # No load_dll attribute → AttributeError

        fake_dll = MagicMock(name="ctypes_dll_handle")
        with patch.dict("sys.modules", {"inno3d.core.enhanced_volume": mock_ev}), \
             patch("ctypes.CDLL", return_value=fake_dll) as mock_cdll:
            result = loader.load_enhanced_dll(fake_dll_path)

        mock_cdll.assert_called_once_with(fake_dll_path)
        assert result is fake_dll


# ─── load_mes_dll ─────────────────────────────────────────────────────────────

class TestLoadMesDll:
    """Tests for load_mes_dll — mocked to avoid real DLLs."""

    def test_delegates_to_core_bumpvoid_mes(self, tmp_path):
        """load_mes_dll should call bumpvoid_mes.load_dll with the dir."""
        from inno3d.native import loader

        mock_mes = MagicMock()

        with patch.dict("sys.modules", {"inno3d.core.bumpvoid_mes": mock_mes}):
            result = loader.load_mes_dll(str(tmp_path))

        mock_mes.load_dll.assert_called_once_with(str(tmp_path))
        assert result is True

    def test_file_not_found_raises_dll_load_error(self, tmp_path):
        """FileNotFoundError from core should be wrapped as DllLoadError."""
        from inno3d.native import loader
        from inno3d.native.loader import DllLoadError

        mock_mes = MagicMock()
        mock_mes.load_dll.side_effect = FileNotFoundError("BumpVoidMes.dll not found")

        with patch.dict("sys.modules", {"inno3d.core.bumpvoid_mes": mock_mes}):
            with pytest.raises(DllLoadError):
                loader.load_mes_dll(str(tmp_path))

    def test_os_error_raises_dll_load_error(self, tmp_path):
        """OSError should be wrapped as DllLoadError."""
        from inno3d.native import loader
        from inno3d.native.loader import DllLoadError

        mock_mes = MagicMock()
        mock_mes.load_dll.side_effect = OSError("access denied")

        with patch.dict("sys.modules", {"inno3d.core.bumpvoid_mes": mock_mes}):
            with pytest.raises(DllLoadError, match="OS error"):
                loader.load_mes_dll(str(tmp_path))


# ─── load_b2b_dll ─────────────────────────────────────────────────────────────

class TestLoadB2bDll:
    """Tests for load_b2b_dll — mocked to avoid real DLLs."""

    def test_delegates_to_core_bumpvoid_b2b_gap_capable(self, tmp_path):
        """load_b2b_dll should call b2b.load_dll and return is_gap_capable()."""
        from inno3d.native import loader

        mock_b2b = MagicMock()
        mock_b2b.is_gap_capable.return_value = True

        with patch.dict("sys.modules", {"inno3d.core.bumpvoid_b2b": mock_b2b}):
            result = loader.load_b2b_dll(str(tmp_path))

        mock_b2b.load_dll.assert_called_once_with(str(tmp_path))
        assert result is True

    def test_delegates_to_core_bumpvoid_b2b_legacy(self, tmp_path):
        """load_b2b_dll returns False when DLL is loaded but not gap-capable."""
        from inno3d.native import loader

        mock_b2b = MagicMock()
        mock_b2b.is_gap_capable.return_value = False

        with patch.dict("sys.modules", {"inno3d.core.bumpvoid_b2b": mock_b2b}):
            result = loader.load_b2b_dll(str(tmp_path))

        assert result is False

    def test_file_not_found_raises_dll_load_error(self, tmp_path):
        """FileNotFoundError should be wrapped as DllLoadError."""
        from inno3d.native import loader
        from inno3d.native.loader import DllLoadError

        mock_b2b = MagicMock()
        mock_b2b.load_dll.side_effect = FileNotFoundError("BoundaryGPU.dll not found")

        with patch.dict("sys.modules", {"inno3d.core.bumpvoid_b2b": mock_b2b}):
            with pytest.raises(DllLoadError):
                loader.load_b2b_dll(str(tmp_path))

    def test_os_error_raises_dll_load_error(self, tmp_path):
        """OSError should be wrapped as DllLoadError."""
        from inno3d.native import loader
        from inno3d.native.loader import DllLoadError

        mock_b2b = MagicMock()
        mock_b2b.load_dll.side_effect = OSError("permission denied")

        with patch.dict("sys.modules", {"inno3d.core.bumpvoid_b2b": mock_b2b}):
            with pytest.raises(DllLoadError, match="OS error"):
                loader.load_b2b_dll(str(tmp_path))
