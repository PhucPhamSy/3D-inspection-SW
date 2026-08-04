"""Wall-time instrumentation for Online mode volume load → first MPR display.

Writes a human-readable report under ``output/`` at repo root (best-effort).
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, Optional

from inno3d.infra.paths import project_root


def _fmt_seconds(seconds: float) -> str:
    return f"{seconds:.4f} s ({seconds * 1000.0:.2f} ms)"


class OnlineMprWalltimeReport:
    """Collect phase durations for one Online load→MPR run."""

    def __init__(
        self,
        *,
        host_path: str = "",
        load_path: str = "",
        load_note: str = "",
    ) -> None:
        self._lock = threading.Lock()
        self._t0 = time.perf_counter()
        self._load_thread_t0: Optional[float] = None
        self._durations: Dict[str, float] = {}
        self._marks: Dict[str, float] = {}
        self._meta: Dict[str, Any] = {
            "host_path": host_path,
            "load_path": load_path,
            "load_note": load_note,
            "crop_used": False,
            "lve_enabled": False,
        }

    def set_meta(self, key: str, value: Any) -> None:
        with self._lock:
            self._meta[key] = value

    def mark(self, name: str) -> None:
        """Record elapsed offset from pipeline start."""
        offset = time.perf_counter() - self._t0
        with self._lock:
            self._marks[name] = offset

    def get_mark(self, name: str) -> Optional[float]:
        with self._lock:
            return self._marks.get(name)

    def record(self, name: str, duration_s: float) -> None:
        with self._lock:
            self._durations[name] = duration_s

    def get_duration(self, name: str) -> Optional[float]:
        with self._lock:
            return self._durations.get(name)

    def begin_load_thread(self) -> None:
        self._load_thread_t0 = time.perf_counter()
        self.mark("load_thread_start")

    def end_load_thread_callback(self, decode_s: Optional[float] = None) -> None:
        if decode_s is not None:
            self.record("load_volume_thread.decode", decode_s)
        if self._load_thread_t0 is not None:
            self.record(
                "load_volume_thread.callback_wall",
                time.perf_counter() - self._load_thread_t0,
            )

    def elapsed_total(self) -> float:
        return time.perf_counter() - self._t0

    def write_report(self, output_dir: Optional[Path] = None) -> Optional[Path]:
        """Best-effort write; returns path or None on failure."""
        try:
            out_dir = output_dir or (project_root() / "output")
            out_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            hint = ""
            load_path = str(self._meta.get("load_path") or "")
            if load_path:
                hint_part = Path(load_path).stem[:24].replace(" ", "_")
                if hint_part:
                    hint = f"_{hint_part}"
            path = out_dir / f"online_mpr_walltime_{ts}{hint}.txt"
            path.write_text("\n".join(self._format_lines()) + "\n", encoding="utf-8")
            return path
        except Exception as e:
            print(f"[ONLINE WALLTIME] report write failed: {e}")
            return None

    def _format_lines(self) -> list[str]:
        with self._lock:
            durations = dict(self._durations)
            marks = dict(self._marks)
            meta = dict(self._meta)
        total = self.elapsed_total()
        lines = [
            "Online mode: volume load → first MPR wall-time report",
            f"Generated: {datetime.now().isoformat(timespec='seconds')}",
            "",
            "--- Context ---",
        ]
        for key in (
            "host_path",
            "load_path",
            "load_note",
            "crop_used",
            "lve_enabled",
            "lve_used_store",
            "volume_shape",
            "volume_dtype",
            "volume_nbytes",
            "error",
        ):
            if key in meta and meta[key] is not None:
                lines.append(f"{key}: {meta[key]}")
        lines.extend(["", "--- Milestones (offset from pipeline start) ---"])
        milestone_order = (
            "pipeline_start",
            "load_thread_start",
            "on_online_volume_loaded.start",
            "first_mpr_ready",
        )
        seen_m = set()
        for name in milestone_order:
            if name in marks:
                lines.append(f"{name}: {_fmt_seconds(marks[name])}")
                seen_m.add(name)
        for name in sorted(marks.keys()):
            if name not in seen_m:
                lines.append(f"{name}: {_fmt_seconds(marks[name])}")
        lines.extend(["", "--- Phase durations ---"])
        phase_order = [
            "roi_crop",
            "crop_volume_for_online_total",
            "load_volume_thread.decode",
            "load_volume_thread.callback_wall",
            "load_volume_thread_total",
            "on_online_volume_loaded.handler",
            "on_online_volume_loaded.start",
            "_on_online_volume_loaded_start",
            "on_volume_loaded.total",
            "store_wrap",
            "minmax",
            "wl_setup",
            "update_all_views.total",
            "on_volume_loaded.update_all_views_total",
            "update_all_views.from_on_volume_loaded",
            "render_slice.axial",
            "render_slice.coronal",
            "render_slice.sagittal",
            "render_slice.coronal_deferred",
            "render_slice.sagittal_deferred",
            "render_slice_axial",
            "render_slice_coronal",
            "render_slice_sagittal",
            "receive_to_first_mpr_ready",
        ]
        seen_d = set()
        for name in phase_order:
            if name in durations:
                lines.append(f"{name}: {_fmt_seconds(durations[name])}")
                seen_d.add(name)
        for name in sorted(durations.keys()):
            if name not in seen_d:
                lines.append(f"{name}: {_fmt_seconds(durations[name])}")
        lines.extend(
            [
                "",
                "--- End-to-end ---",
                (
                    f"receive_to_first_mpr_ready: "
                    f"{_fmt_seconds(float(marks['first_mpr_ready']))}"
                    if "first_mpr_ready" in marks
                    else f"receive_to_report_write: {_fmt_seconds(total)}"
                ),
            ]
        )
        if "first_mpr_ready" in marks:
            lines.append(
                f"report_write_elapsed: {_fmt_seconds(total)} "
                f"(includes work after first MPR; not first-paint latency)"
            )
        return lines


@contextmanager
def online_walltime_phase(
    report: Optional[OnlineMprWalltimeReport], phase: str
) -> Generator[None, None, None]:
    """Time a block and record duration on *report* when present."""
    if report is None:
        yield
        return
    start = time.perf_counter()
    try:
        yield
    finally:
        report.record(phase, time.perf_counter() - start)
