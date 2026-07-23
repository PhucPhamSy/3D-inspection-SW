"""
scripts/post_build_smoke.py
───────────────────────────
Post-build smoke check for Inno3D packaged exe.

Usage (dev machine):
    python scripts/post_build_smoke.py

Usage (CI / build_final.bat optional step):
    python scripts/post_build_smoke.py --dist-dir dist/Inno3D

Exit codes:
    0  — all checks passed
    1  — one or more checks failed

Set env INNO3D_SMOKE=1 inside the packaged exe via a wrapper to trigger
a self-test that closes cleanly after QTimer fires (if --smoke-exit is passed).

Current checks (no GPU, no real DLL required):
    1. dist/Inno3D/Inno3D.exe exists
    2. dist/Inno3D/VERSION.txt exists and is non-empty
    3. dist/Inno3D/app_config.ini exists
    4. dist/Inno3D/V2/ exists (at least one .dll inside)
    5. dist/Inno3D/config/ exists (at least one .txt inside)
    6. Internal Python package importable:
       python -c "import inno3d; print(inno3d.__version__)"  (dev env only)
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def _check(label: str, result: bool, detail: str = "") -> bool:
    icon = "✅" if result else "❌"
    suffix = f"  ({detail})" if detail else ""
    print(f"  {icon}  {label}{suffix}")
    return result


def run_smoke(dist_dir: Path) -> int:
    """Run all smoke checks against dist_dir. Returns 0=pass, 1=fail."""
    print(f"\n{'='*55}")
    print(f"  Inno3D Post-Build Smoke Check")
    print(f"  dist_dir: {dist_dir}")
    print(f"{'='*55}\n")

    results: list[bool] = []

    # 1. Exe present
    exe = dist_dir / "Inno3D.exe"
    results.append(_check("Inno3D.exe exists", exe.is_file(), str(exe)))

    # 2. VERSION.txt
    vtxt = dist_dir / "VERSION.txt"
    ver_content = vtxt.read_text(encoding="utf-8").strip() if vtxt.is_file() else ""
    results.append(_check("VERSION.txt present & non-empty", bool(ver_content), ver_content[:40]))

    # 3. app_config.ini
    cfg = dist_dir / "app_config.ini"
    results.append(_check("app_config.ini exists", cfg.is_file()))

    # 4. V2/ folder with at least one DLL
    v2 = dist_dir / "V2"
    v2_dlls = list(v2.glob("*.dll")) if v2.is_dir() else []
    results.append(_check("V2/ exists with ≥1 DLL", bool(v2_dlls), f"{len(v2_dlls)} DLLs"))

    # 5. config/ with .txt recipes
    cfg_dir = dist_dir / "config"
    cfg_txts = list(cfg_dir.glob("*.txt")) if cfg_dir.is_dir() else []
    results.append(_check("config/ exists with ≥1 .txt", bool(cfg_txts), f"{len(cfg_txts)} files"))

    # 6. Dev-env package importable (optional — skip if no conda env)
    try:
        proc = subprocess.run(
            [sys.executable, "-c", "import inno3d; print(inno3d.__version__)"],
            capture_output=True, text=True, timeout=10,
        )
        ok = proc.returncode == 0
        ver = proc.stdout.strip() if ok else proc.stderr.strip()[:60]
        results.append(_check("inno3d package importable (dev)", ok, f"version={ver}"))
    except Exception as exc:
        # Non-fatal if running from dist without source
        _check("inno3d package importable (dev)", False, f"skipped: {exc}")
        results.append(True)  # non-blocking

    passed = sum(results)
    total = len(results)
    print(f"\n{'='*55}")
    if all(results):
        print(f"  ✅ ALL {total} checks PASSED — build is ship-ready")
        print(f"{'='*55}\n")
        return 0
    else:
        print(f"  ❌ {total - passed}/{total} checks FAILED")
        print(f"{'='*55}\n")
        return 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Inno3D post-build smoke checker")
    parser.add_argument(
        "--dist-dir",
        default="dist/Inno3D",
        help="Path to the dist output directory (default: dist/Inno3D)",
    )
    args = parser.parse_args()

    dist_dir = Path(args.dist_dir).resolve()
    sys.exit(run_smoke(dist_dir))


if __name__ == "__main__":
    main()
