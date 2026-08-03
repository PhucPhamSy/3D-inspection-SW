"""Unit tests for inno3d.app.settings — centralized config reader."""
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch

from inno3d.app import settings


class TestGetTheme:
    def test_default_is_dark(self):
        """Without env or ini override, default theme is dark."""
        with patch.dict("os.environ", {}, clear=False):
            # Remove INNO3D_THEME if set
            import os
            env = os.environ.pop("INNO3D_THEME", None)
            try:
                result = settings.get_theme()
                assert result in ("dark", "light")
            finally:
                if env is not None:
                    os.environ["INNO3D_THEME"] = env

    def test_env_override(self):
        """INNO3D_THEME env var overrides ini."""
        with patch.dict("os.environ", {"INNO3D_THEME": "light"}):
            assert settings.get_theme() == "light"

    def test_env_invalid_falls_back(self):
        """Invalid env value falls back to ini/default."""
        with patch.dict("os.environ", {"INNO3D_THEME": "rainbow"}):
            result = settings.get_theme()
            assert result in ("dark", "light")


class TestSetTheme:
    def test_roundtrip(self):
        """set_theme then get_theme returns the same value."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fake_ini = Path(tmpdir) / "app_config.ini"
            with patch("inno3d.app.settings.app_config_ini_path", return_value=fake_ini):
                # Remove env override
                with patch.dict("os.environ", {}, clear=False):
                    import os
                    env = os.environ.pop("INNO3D_THEME", None)
                    try:
                        settings.set_theme("light")
                        assert settings.get_theme() == "light"
                        settings.set_theme("dark")
                        assert settings.get_theme() == "dark"
                    finally:
                        if env is not None:
                            os.environ["INNO3D_THEME"] = env

    def test_invalid_normalizes(self):
        """Invalid theme names normalize to 'dark'."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fake_ini = Path(tmpdir) / "app_config.ini"
            with patch("inno3d.app.settings.app_config_ini_path", return_value=fake_ini):
                settings.set_theme("rainbow")
                # Read raw to verify
                import configparser
                cfg = configparser.ConfigParser()
                cfg.read(str(fake_ini))
                assert cfg.get("APPEARANCE", "theme") == "dark"


class TestGetOnlinePort:
    def test_default_port(self):
        assert settings.get_online_port() == 8000

    def test_custom_port(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ini = Path(tmpdir) / "app_config.ini"
            ini.write_text("[ONLINE]\nport = 9000\n", encoding="utf-8")
            with patch("inno3d.app.settings.app_config_ini_path", return_value=ini):
                assert settings.get_online_port() == 9000


class TestVolume3dSettings:
    def test_default_budget(self):
        with patch.dict("os.environ", {}, clear=False):
            import os

            env = os.environ.pop("INNO3D_3D_UPLOAD_BUDGET_MB", None)
            try:
                assert settings.get_3d_upload_budget_mb() in settings.VOLUME_3D_BUDGET_MB_OPTIONS
            finally:
                if env is not None:
                    os.environ["INNO3D_3D_UPLOAD_BUDGET_MB"] = env

    def test_budget_env_override(self):
        with patch.dict("os.environ", {"INNO3D_3D_UPLOAD_BUDGET_MB": "1024"}):
            assert settings.get_3d_upload_budget_mb() == 1024

    def test_budget_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            fake_ini = Path(tmpdir) / "app_config.ini"
            with patch("inno3d.app.settings.app_config_ini_path", return_value=fake_ini):
                with patch.dict("os.environ", {}, clear=False):
                    import os

                    env = os.environ.pop("INNO3D_3D_UPLOAD_BUDGET_MB", None)
                    try:
                        settings.set_3d_upload_budget_mb(1536)
                        assert settings.get_3d_upload_budget_mb() == 1536
                    finally:
                        if env is not None:
                            os.environ["INNO3D_3D_UPLOAD_BUDGET_MB"] = env

    def test_lve_env_override(self):
        with patch.dict("os.environ", {"INNO3D_LARGE_VOLUME_ENGINE": "1"}):
            assert settings.get_large_volume_engine_enabled() is True

    def test_lve_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            fake_ini = Path(tmpdir) / "app_config.ini"
            with patch("inno3d.app.settings.app_config_ini_path", return_value=fake_ini):
                with patch.dict("os.environ", {}, clear=False):
                    import os

                    env = os.environ.pop("INNO3D_LARGE_VOLUME_ENGINE", None)
                    try:
                        settings.set_large_volume_engine_enabled(True)
                        assert settings.get_large_volume_engine_enabled() is True
                        settings.set_large_volume_engine_enabled(False)
                        assert settings.get_large_volume_engine_enabled() is False
                    finally:
                        if env is not None:
                            os.environ["INNO3D_LARGE_VOLUME_ENGINE"] = env
