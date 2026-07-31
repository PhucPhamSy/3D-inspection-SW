"""Unit tests for inno3d.core.perf_timing — env-gated timing helpers."""
import pytest

from inno3d.core.perf_timing import perf_enabled, perf_phase, perf_timed


class TestPerfEnabled:
    def test_disabled_by_default(self, monkeypatch):
        monkeypatch.delenv("INNO3D_PERF", raising=False)
        assert perf_enabled() is False

    def test_enabled_when_inno3d_perf_is_one(self, monkeypatch):
        monkeypatch.setenv("INNO3D_PERF", "1")
        assert perf_enabled() is True


class TestDisabledByDefault:
    def test_perf_phase_no_output_and_no_crash(self, capsys, monkeypatch):
        monkeypatch.delenv("INNO3D_PERF", raising=False)
        with perf_phase("silent_phase", detail="ignored"):
            pass
        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == ""

    def test_perf_timed_no_output_and_no_crash(self, capsys, monkeypatch):
        monkeypatch.delenv("INNO3D_PERF", raising=False)

        @perf_timed("silent_fn")
        def sample():
            return 42

        assert sample() == 42
        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == ""


class TestPerfPhaseEnabled:
    def test_emits_timing_with_detail(self, capsys, monkeypatch):
        monkeypatch.setenv("INNO3D_PERF", "1")
        with perf_phase("load_volume", detail="512^3"):
            pass
        captured = capsys.readouterr()
        assert "[PERF] load_volume:" in captured.out
        assert "ms | 512^3" in captured.out

    def test_emits_timing_without_detail(self, capsys, monkeypatch):
        monkeypatch.setenv("INNO3D_PERF", "1")
        with perf_phase("simple"):
            pass
        captured = capsys.readouterr()
        assert "[PERF] simple:" in captured.out
        assert " ms" in captured.out
        assert " | " not in captured.out.split("[PERF]")[1]


class TestPerfTimedDecorator:
    def test_decorator_emits_timing(self, capsys, monkeypatch):
        monkeypatch.setenv("INNO3D_PERF", "1")

        @perf_timed("my_func")
        def my_func(x):
            return x * 2

        assert my_func(3) == 6
        captured = capsys.readouterr()
        assert "[PERF] my_func:" in captured.out
        assert " ms" in captured.out

    def test_decorator_uses_function_name_when_phase_omitted(self, capsys, monkeypatch):
        monkeypatch.setenv("INNO3D_PERF", "1")

        @perf_timed()
        def compute():
            return "ok"

        assert compute() == "ok"
        captured = capsys.readouterr()
        assert "[PERF] compute:" in captured.out
