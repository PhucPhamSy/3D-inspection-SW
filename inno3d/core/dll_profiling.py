"""
Shared DLL timing / bottleneck profiling helpers.

Each ISP DLL exposes optional (ABI-additive) exports:
  Xxx_SetProfiling(int enable)
  Xxx_GetProfiling() -> int
  Xxx_GetLastTiming(TimingDetail*) -> int

Default is OFF so production runs keep zero stage-timer overhead.
Coarse fields (SEG bumpTime/voidTime, MES elapsedSec) remain always-on.
"""

from __future__ import annotations

import ctypes
import os
import platform
import subprocess
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

PROFILING_REPORT_FILENAME = "dll_profiling.txt"

# Inline hints (English) appended next to each timing line in reports/logs.
PROFILING_INLINE_HINTS: Dict[str, str] = {
    "load_io": "read image/volume files from disk into RAM",
    "save_io": "write masks, CSV, or TIFF results from RAM to disk",
    "gpu_h2d": "Host-to-Device — copy data from CPU RAM to GPU VRAM",
    "gpu_d2h": "Device-to-Host — copy results from GPU VRAM back to CPU RAM",
    "h2d": "Host-to-Device — upload CPU data to GPU",
    "d2h": "Device-to-Host — download GPU results to CPU",
    "bump_algo": "bump segmentation algorithm (GPU/CPU compute)",
    "void_algo": "void segmentation algorithm",
    "clean_air": "clean-air artifact filtering / post-processing",
    "prep": "mask prep — relabel, layer assignment, formatting",
    "ccl": "Connected Component Labeling — assign object IDs",
    "accumulate": "accumulate object stats (volume, bbox, centroid, …)",
    "far": "FAR — false-alarm rejection / invalid object filter",
    "other/csv": "CSV export and remaining overhead",
    "extract": "extract boundary/contour from mask on GPU",
    "host_group": "group boundaries on CPU after GPU stage",
    "pair_dist": "compute bump-to-bump gap distances (B2B)",
    "csv_io": "write boundary/gap CSV files",
    "infer": "ONNX/TensorRT GPU inference (enhancement)",
    "pad/convert": "pad / dtype / layout conversion around inference",
    "slices": "number of enhanced slices (batch info, not a timer)",
    "total": "total stage wall time (100% baseline for percentages)",
    "algo": "total algorithm compute time (excluding main I/O)",
    "other": "other overhead not counted in main I/O or algo buckets",
}

PROFILING_GLOSSARY_LINES: List[str] = [
    "Glossary:",
    "  load_io / save_io     : disk read / write time",
    "  h2d / gpu_h2d         : Host-to-Device — CPU RAM → GPU VRAM upload",
    "  d2h / gpu_d2h         : Device-to-Host — GPU VRAM → CPU RAM download",
    "  bump_algo / void_algo : bump and void segmentation compute",
    "  ccl                   : connected-component labeling (object IDs)",
    "  far                   : false-alarm rejection",
    "  pair_dist             : bump-to-bump gap distance (B2B)",
    "  infer                 : AI enhancement inference (ONNX/TensorRT)",
    "  %                     : share of that stage TOTAL time",
    "  [subset of algo]      : GPU copy time counted inside algo, not added to TOTAL",
    "  TFLOPS note           : peak FP32 is a hardware spec estimate; per-kernel FLOPs are not counted",
]

# Vendor peak FP32 TFLOPS (approx.) for common GPUs when nvidia-smi cannot derive cores.
_GPU_PEAK_FP32_TFLOPS: Dict[str, float] = {
    "RTX 4090": 82.6,
    "RTX 4080": 48.7,
    "RTX 4070": 29.1,
    "RTX 3070 TI": 21.7,
    "RTX 3070": 20.3,
    "RTX 3090": 35.6,
    "RTX 3080": 29.8,
    "RTX A6000": 38.7,
    "RTX A5000": 27.8,
    "RTX A4500": 23.7,
    "RTX A4000": 19.2,
    "A100": 19.5,
    "A100 80GB": 19.5,
    "H100": 67.0,
    "Tesla T4": 8.1,
    "Quadro RTX 6000": 16.3,
    "Quadro RTX 8000": 16.3,
}


class IspTimingDetail(ctypes.Structure):
    """Layout shared by SEG / MES / B2B / ENH timing structs."""

    _fields_ = [
        ("enabled", ctypes.c_int),
        ("totalSec", ctypes.c_double),
        ("loadIoSec", ctypes.c_double),
        ("saveIoSec", ctypes.c_double),
        ("h2dSec", ctypes.c_double),
        ("d2hSec", ctypes.c_double),
        ("algoSec", ctypes.c_double),
        ("otherSec", ctypes.c_double),
        ("stage0Sec", ctypes.c_double),
        ("stage1Sec", ctypes.c_double),
        ("stage2Sec", ctypes.c_double),
        ("stage3Sec", ctypes.c_double),
        ("report", ctypes.c_char * 2048),
    ]


def timing_to_dict(detail: IspTimingDetail) -> Dict[str, Any]:
    report = ""
    try:
        report = detail.report.decode("utf-8", errors="replace").rstrip("\x00")
    except Exception:
        report = ""
    return {
        "enabled": bool(detail.enabled),
        "total_sec": float(detail.totalSec),
        "load_io_sec": float(detail.loadIoSec),
        "save_io_sec": float(detail.saveIoSec),
        "h2d_sec": float(detail.h2dSec),
        "d2h_sec": float(detail.d2hSec),
        "algo_sec": float(detail.algoSec),
        "other_sec": float(detail.otherSec),
        "stage0_sec": float(detail.stage0Sec),
        "stage1_sec": float(detail.stage1Sec),
        "stage2_sec": float(detail.stage2Sec),
        "stage3_sec": float(detail.stage3Sec),
        "report": report,
    }


def _bind_optional(dll, set_name: str, get_name: str, timing_name: str) -> bool:
    if not hasattr(dll, set_name) or not hasattr(dll, timing_name):
        return False
    getattr(dll, set_name).argtypes = [ctypes.c_int]
    getattr(dll, set_name).restype = None
    if hasattr(dll, get_name):
        getattr(dll, get_name).argtypes = []
        getattr(dll, get_name).restype = ctypes.c_int
    getattr(dll, timing_name).argtypes = [ctypes.POINTER(IspTimingDetail)]
    getattr(dll, timing_name).restype = ctypes.c_int
    return True


def set_profiling(dll, enable: bool, set_name: str) -> bool:
    """Return True if DLL supports profiling and flag was applied."""
    if dll is None or not hasattr(dll, set_name):
        return False
    try:
        getattr(dll, set_name)(1 if enable else 0)
        return True
    except Exception:
        return False


def get_last_timing(dll, timing_name: str) -> Optional[Dict[str, Any]]:
    if dll is None or not hasattr(dll, timing_name):
        return None
    detail = IspTimingDetail()
    try:
        ok = getattr(dll, timing_name)(ctypes.byref(detail))
    except Exception:
        return None
    if not ok and not detail.enabled:
        return None
    return timing_to_dict(detail)


def _field_key_from_line(line: str) -> Optional[str]:
    if ":" not in line:
        return None
    return line.split(":", 1)[0].strip().lower()


def annotate_profiling_report(text: str) -> str:
    """Append English hints next to known profiling field names."""
    if not text:
        return text
    out: List[str] = []
    for line in text.splitlines():
        if not line.strip():
            out.append(line)
            continue
        if line.lstrip().startswith("[") and "]" in line.split(":", 1)[0]:
            out.append(line)
            continue
        if "#" in line:
            out.append(line)
            continue
        key = _field_key_from_line(line)
        hint = PROFILING_INLINE_HINTS.get(key or "")
        if hint:
            out.append(f"{line.rstrip()}  # {hint}")
        else:
            out.append(line)
    return "\n".join(out)


def format_timing_banner(module: str, info: Optional[Dict[str, Any]]) -> str:
    if not info:
        return f"[{module}] Profiling: (no timing / DLL not rebuilt yet)"
    report = (info.get("report") or "").strip()
    if report:
        return annotate_profiling_report(report)
    fallback = (
        f"[{module} Profiling]\n"
        f"  load_io : {info.get('load_io_sec', 0):.3f}s\n"
        f"  algo    : {info.get('algo_sec', 0):.3f}s\n"
        f"  save_io : {info.get('save_io_sec', 0):.3f}s\n"
        f"  h2d     : {info.get('h2d_sec', 0):.3f}s\n"
        f"  d2h     : {info.get('d2h_sec', 0):.3f}s\n"
        f"  TOTAL   : {info.get('total_sec', 0):.3f}s"
    )
    return annotate_profiling_report(fallback)


def _human_bytes(nbytes: int) -> str:
    if nbytes >= 1024 ** 3:
        return f"{nbytes / (1024 ** 3):.2f} GiB"
    if nbytes >= 1024 ** 2:
        return f"{nbytes / (1024 ** 2):.1f} MiB"
    if nbytes >= 1024:
        return f"{nbytes / 1024:.1f} KiB"
    return f"{nbytes} B"


def _format_volume_lines(
    volume: Any,
    spacing: Optional[Sequence[float]] = None,
    label: str = "Input volume shape (Z x Y x X)",
) -> List[str]:
    if volume is None:
        return [f"{label}: (not loaded)"]
    try:
        import numpy as np

        arr = np.asarray(volume)
        if arr.ndim == 2:
            shape = (1, int(arr.shape[0]), int(arr.shape[1]))
        elif arr.ndim >= 3:
            shape = tuple(int(x) for x in arr.shape[:3])
        else:
            return [f"{label}: (unsupported ndim={arr.ndim})"]
        nbytes = int(arr.nbytes)
        lines = [
            f"{label}: {shape[0]} x {shape[1]} x {shape[2]}",
            f"  dtype: {arr.dtype}  |  size in RAM: {_human_bytes(nbytes)}  |  voxels: {int(np.prod(shape)):,}",
        ]
        if spacing is not None:
            try:
                sp = [float(spacing[i]) for i in range(min(3, len(spacing)))]
                while len(sp) < 3:
                    sp.append(sp[-1] if sp else 1.0)
                phys_z = shape[0] * sp[0]
                phys_y = shape[1] * sp[1]
                phys_x = shape[2] * sp[2]
                lines.append(
                    f"  voxel spacing (Z,Y,X): {sp[0]:g}, {sp[1]:g}, {sp[2]:g}  "
                    f"|  physical extent (Z,Y,X): {phys_z:g}, {phys_y:g}, {phys_x:g}"
                )
            except Exception:
                pass
        return lines
    except Exception as exc:
        return [f"{label}: (error: {exc})"]


def _run_nvidia_smi_query(args: List[str], timeout: float = 2.0) -> Optional[str]:
    try:
        return subprocess.check_output(
            ["nvidia-smi", *args],
            creationflags=subprocess.CREATE_NO_WINDOW
            if hasattr(subprocess, "CREATE_NO_WINDOW")
            else 0,
            timeout=timeout,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def _lookup_peak_tflops(gpu_name: str) -> Optional[float]:
    name_upper = (gpu_name or "").upper()
    for key, val in _GPU_PEAK_FP32_TFLOPS.items():
        if key.upper() in name_upper:
            return val
    return None


def _estimate_tflops_line(
    gpu_name: str,
    cuda_cores: Optional[int],
    max_sm_mhz: Optional[float],
) -> str:
    if cuda_cores and max_sm_mhz and cuda_cores > 0 and max_sm_mhz > 0:
        est = cuda_cores * (max_sm_mhz / 1000.0) * 2.0 / 1000.0
        return (
            f"  peak FP32 throughput (est.): ~{est:.1f} TFLOPS "
            f"({cuda_cores} CUDA cores @ {max_sm_mhz:.0f} MHz SM clock, 2 FLOP/cycle/core)"
        )
    spec = _lookup_peak_tflops(gpu_name)
    if spec is not None:
        return f"  peak FP32 throughput (vendor spec): ~{spec:.1f} TFLOPS"
    return "  peak FP32 throughput: not available (GPU model not in lookup; per-kernel FLOPs are not measured)"


def _collect_gpu_lines(device_id: int = 0) -> List[str]:
    lines = ["GPU:"]
    query = _run_nvidia_smi_query([
        f"--id={device_id}",
        "--query-gpu=index,name,uuid,driver_version,memory.total,memory.used,memory.free,"
        "utilization.gpu,utilization.memory,temperature.gpu,power.draw,power.limit,"
        "clocks.current.graphics,clocks.current.sm,clocks.max.graphics,clocks.max.sm,"
        "compute_cap,pcie.link.gen.current,pcie.link.width.current,"
        "pcie.link.gen.max,pcie.link.width.max",
        "--format=csv,noheader,nounits",
    ])
    if not query:
        # SEG DLL fallback when driver tools missing
        try:
            from inno3d.core import bumpvoid

            if bumpvoid.has_cuda():
                vram_mb = bumpvoid.get_gpu_memory_mb()
                lines.append(f"  CUDA (SEG DLL): available  |  VRAM total (driver query): {vram_mb} MiB")
            else:
                lines.append("  CUDA (SEG DLL): not available / no NVIDIA GPU detected")
        except Exception:
            lines.append("  nvidia-smi not available — install NVIDIA driver for GPU details")
        lines.append("  note: TFLOPS/FLOPs are not counted at runtime; use h2d/algo/d2h timers for bottlenecks")
        return lines

    parts = [p.strip() for p in query.split(",")]
    fields = [
        "index", "name", "uuid", "driver", "mem_total_mb", "mem_used_mb", "mem_free_mb",
        "util_gpu_pct", "util_mem_pct", "temp_c", "power_draw_w", "power_limit_w",
        "clock_gfx_mhz", "clock_sm_mhz", "clock_gfx_max_mhz", "clock_sm_max_mhz",
        "compute_cap", "pcie_gen_cur", "pcie_width_cur", "pcie_gen_max", "pcie_width_max",
    ]
    info = dict(zip(fields, parts + [""] * max(0, len(fields) - len(parts))))

    gpu_name = info.get("name") or "unknown"
    lines.append(f"  device index: {info.get('index', device_id)}")
    lines.append(f"  name: {gpu_name}")
    lines.append(f"  driver: {info.get('driver', '?')}  |  compute capability: {info.get('compute_cap', '?')}")
    lines.append(
        f"  VRAM total / used / free: {info.get('mem_total_mb', '?')} / "
        f"{info.get('mem_used_mb', '?')} / {info.get('mem_free_mb', '?')} MiB"
    )
    lines.append(
        f"  utilization GPU / VRAM (snapshot): {info.get('util_gpu_pct', '?')}% / "
        f"{info.get('util_mem_pct', '?')}%"
    )
    lines.append(
        f"  temperature: {info.get('temp_c', '?')} C  |  power draw / limit: "
        f"{info.get('power_draw_w', '?')} / {info.get('power_limit_w', '?')} W"
    )
    lines.append(
        f"  clocks SM current / max: {info.get('clock_sm_mhz', '?')} / "
        f"{info.get('clock_sm_max_mhz', '?')} MHz  |  graphics current / max: "
        f"{info.get('clock_gfx_mhz', '?')} / {info.get('clock_gfx_max_mhz', '?')} MHz"
    )
    lines.append(
        f"  PCIe current / max: Gen{info.get('pcie_gen_cur', '?')} x{info.get('pcie_width_cur', '?')}  /  "
        f"Gen{info.get('pcie_gen_max', '?')} x{info.get('pcie_width_max', '?')}"
    )

    cuda_cores: Optional[int] = None
    max_sm: Optional[float] = None
    try:
        max_sm = float(info.get("clock_sm_max_mhz") or 0) or None
    except Exception:
        max_sm = None
    detail = _run_nvidia_smi_query(["-q", "-d", "CLOCK,COMPUTE,MEMORY,UTILIZATION,POWER,PCI"])
    if detail:
        for raw in detail.splitlines():
            s = raw.strip()
            if "CUDA Cores" in s and ":" in s:
                try:
                    cuda_cores = int(s.split(":", 1)[1].strip())
                except Exception:
                    pass
    lines.append(_estimate_tflops_line(gpu_name, cuda_cores, max_sm))

    try:
        from inno3d.core import bumpvoid

        if bumpvoid.has_cuda():
            lines.append(f"  SEG DLL CUDA: yes  |  SEG DLL VRAM query: {bumpvoid.get_gpu_memory_mb()} MiB total")
        else:
            lines.append("  SEG DLL CUDA: no")
    except Exception:
        pass

    lines.append("  note: runtime FLOP count is not instrumented; use h2d / algo / d2h splits for bottlenecks")
    return lines


def _collect_cpu_lines() -> List[str]:
    lines = ["CPU:"]
    try:
        import psutil

        proc = psutil.Process()
        phys = psutil.cpu_count(logical=False) or 0
        logical = psutil.cpu_count(logical=True) or 0
        affinity_count = len(proc.cpu_affinity()) if hasattr(proc, "cpu_affinity") else logical
        threads = proc.num_threads()
        cpu_pct = psutil.cpu_percent(interval=0.05)
        ram = psutil.virtual_memory()

        lines.append(f"  host: {platform.system()} {platform.release()} ({platform.machine()})")
        lines.append(f"  processor: {platform.processor() or 'unknown'}")
        lines.append(f"  physical cores: {phys}  |  logical processors: {logical}")
        lines.append(
            f"  process CPU affinity: {affinity_count}/{logical} logical processors available to this app"
        )
        lines.append(f"  process threads (this Python app): {threads}")
        lines.append(f"  system CPU utilization (snapshot): {cpu_pct:.1f}%")
        lines.append(
            f"  system RAM total / used: {_human_bytes(int(ram.total))} / "
            f"{ram.percent:.1f}% used ({_human_bytes(int(ram.used))})"
        )

        omp = os.environ.get("OMP_NUM_THREADS")
        ocv = os.environ.get("OPENCV_NUM_THREADS")
        lines.append(f"  env OMP_NUM_THREADS: {omp if omp else '(not set — OpenMP uses runtime default)'}")
        lines.append(f"  env OPENCV_NUM_THREADS: {ocv if ocv else '(not set)'}")

        try:
            from inno3d.core import bumpvoid

            seg_omp = bumpvoid.get_thread_count()
            lines.append(f"  OpenMP max threads (SEG DLL BumpVoid_GetThreadCount): {seg_omp}/{logical} logical")
        except Exception:
            lines.append("  OpenMP max threads (SEG DLL): not available (DLL not loaded)")
    except Exception as exc:
        lines.append(f"  (cpu info error: {exc})")
    return lines


def collect_runtime_snapshot_lines(gpu_device_id: int = 0) -> List[str]:
    """Lightweight end-of-run CPU/GPU utilization snapshot."""
    lines = ["End-of-run utilization snapshot:"]
    try:
        import psutil

        proc = psutil.Process()
        lines.append(f"  process threads: {proc.num_threads()}")
        lines.append(f"  system CPU utilization: {psutil.cpu_percent(interval=0.05):.1f}%")
        ram = psutil.virtual_memory()
        lines.append(f"  system RAM used: {ram.percent:.1f}%")
    except Exception as exc:
        lines.append(f"  CPU snapshot error: {exc}")

    q = _run_nvidia_smi_query([
        f"--id={gpu_device_id}",
        "--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu",
        "--format=csv,noheader,nounits",
    ])
    if q:
        p = [x.strip() for x in q.split(",")]
        if len(p) >= 6:
            lines.append(
                f"  GPU util / VRAM util: {p[0]}% / {p[1]}%  |  VRAM used/total: {p[2]}/{p[3]} MiB  "
                f"|  power: {p[4]} W  |  temp: {p[5]} C"
            )
    else:
        lines.append("  GPU snapshot: nvidia-smi not available")
    return lines


def build_profiling_context_lines(
    volume: Any = None,
    spacing: Optional[Sequence[float]] = None,
    gpu_device_id: int = 0,
    extra_lines: Optional[List[str]] = None,
) -> List[str]:
    """Header block: run paths, volume shape, CPU, GPU."""
    lines: List[str] = ["Run context:"]
    if extra_lines:
        lines.extend(extra_lines)
    lines.append("")
    lines.extend(_format_volume_lines(volume, spacing=spacing))
    lines.append("")
    lines.extend(_collect_cpu_lines())
    lines.append("")
    lines.extend(_collect_gpu_lines(device_id=gpu_device_id))
    return lines


class ProfilingReportBuilder:
    """Collect profiling sections for one FOV / inspection run."""

    def __init__(self) -> None:
        self.context_lines: List[str] = []
        self.sections: List[Dict[str, str]] = []
        self.footer_lines: List[str] = []
        self.runtime_snapshot_lines: List[str] = []
        self.gpu_device_id: int = 0

    def reset(self) -> None:
        self.context_lines = []
        self.sections = []
        self.footer_lines = []
        self.runtime_snapshot_lines = []
        self.gpu_device_id = 0

    def set_context(self, lines: List[str], gpu_device_id: int = 0) -> None:
        self.context_lines = [str(x) for x in lines if x is not None]
        self.gpu_device_id = int(gpu_device_id or 0)

    def add_footer(self, lines: List[str]) -> None:
        self.footer_lines.extend(str(x) for x in lines if x)

    def set_runtime_snapshot(self, lines: Optional[List[str]] = None) -> None:
        if lines is None:
            lines = collect_runtime_snapshot_lines(gpu_device_id=self.gpu_device_id)
        self.runtime_snapshot_lines = [str(x) for x in lines if x]

    def add_section(self, module: str, banner: str, note: Optional[str] = None) -> None:
        text = (banner or "").strip()
        if not text:
            return
        self.sections.append(
            {"module": module, "banner": text, "note": (note or "").strip()}
        )

    def render(self) -> str:
        lines = [
            "=" * 72,
            "DLL Profiling Report (bottleneck timing)",
            "=" * 72,
            "",
            *PROFILING_GLOSSARY_LINES,
        ]
        if self.context_lines:
            lines.append("")
            lines.append("-" * 72)
            lines.extend(self.context_lines)
        for sec in self.sections:
            lines.append("")
            lines.append("-" * 72)
            title = f"[{sec['module']}]"
            if sec.get("note"):
                title += f"  ({sec['note']})"
            lines.append(title)
            lines.append("-" * 72)
            lines.append(sec["banner"])
        if self.runtime_snapshot_lines:
            lines.append("")
            lines.append("-" * 72)
            lines.extend(self.runtime_snapshot_lines)
        if self.footer_lines:
            lines.append("")
            lines.append("-" * 72)
            lines.extend(self.footer_lines)
        lines.append("")
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("=" * 72)
        return "\n".join(lines) + "\n"

    def write(self, output_dir: str) -> Optional[str]:
        if not output_dir:
            return None
        if not self.sections and not self.context_lines:
            return None
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, PROFILING_REPORT_FILENAME)
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.render())
        return path
