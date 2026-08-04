"""Unit tests for Online MPR wall-time report helper."""

import time

from inno3d.core.online_mpr_walltime import OnlineMprWalltimeReport, online_walltime_phase


class TestOnlineMprWalltimeReport:
    def test_mark_and_record(self):
        report = OnlineMprWalltimeReport(host_path="test")
        report.mark("pipeline_start")
        report.record("roi_crop", 0.5)
        lines = report._format_lines()
        text = "\n".join(lines)
        assert "pipeline_start" in text
        assert "roi_crop" in text
        assert "host_path: test" in text

    def test_write_report_creates_file(self, tmp_path):
        report = OnlineMprWalltimeReport(host_path="test", load_path="/foo/Input.tif")
        report.mark("pipeline_start")
        out = report.write_report(output_dir=tmp_path)
        assert out is not None
        assert out.exists()
        assert out.name.startswith("online_mpr_walltime_")
        assert "Input" in out.name or out.suffix == ".txt"

    def test_online_walltime_phase_records_duration(self):
        report = OnlineMprWalltimeReport()
        with online_walltime_phase(report, "store_wrap"):
            time.sleep(0.01)
        dur = report.get_duration("store_wrap")
        assert dur is not None
        assert dur >= 0.005

    def test_online_walltime_phase_noop_when_report_none(self):
        with online_walltime_phase(None, "ignored"):
            pass

    def test_deferred_render_phases_are_reported(self):
        report = OnlineMprWalltimeReport()
        report.record("render_slice.axial", 0.01)
        report.record("render_slice.coronal_deferred", 0.02)
        report.record("render_slice.sagittal_deferred", 0.03)
        text = "\n".join(report._format_lines())
        assert "render_slice.axial" in text
        assert "render_slice.coronal_deferred" in text
        assert "render_slice.sagittal_deferred" in text
