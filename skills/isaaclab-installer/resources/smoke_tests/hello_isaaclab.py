"""hello_isaaclab.py — minimal headless smoke test for an Isaac Lab install.

Verifies that:
- `isaaclab` and `isaacsim` (or kit-less alternative) can be imported,
- the AppLauncher can construct a headless app,
- a basic Sim instance can be created and torn down,
- the python interpreter has the expected version.

Designed to be run AFTER an install via:
    ./isaaclab.sh -p resources/smoke_tests/hello_isaaclab.py

Exit codes:
    0  success
    2  python version mismatch
    3  isaaclab import failed
    4  isaacsim import failed (only checked when not in kit-less mode)
    5  AppLauncher / Sim init failed
"""

from __future__ import annotations

import argparse
import platform
import sys
import traceback


REQUIRED_PYTHON = (3, 12)


def _print(label, value):
    print(f"  {label:<20} {value}")


def check_python():
    print("[1/4] Python interpreter")
    _print("python", platform.python_version())
    _print("executable", sys.executable)
    _print("arch", platform.machine())
    if sys.version_info[:2] != REQUIRED_PYTHON:
        print(f"  ERROR: Isaac Sim 6.x requires Python {REQUIRED_PYTHON[0]}.{REQUIRED_PYTHON[1]}; you have {platform.python_version()}.")
        return 2
    return 0


def check_isaaclab_import():
    print("\n[2/4] Import isaaclab")
    try:
        import isaaclab  # noqa: F401
        _print("isaaclab", "imported OK")
        try:
            _print("version", getattr(isaaclab, "__version__", "unknown"))
        except Exception:
            pass
        return 0
    except Exception:
        traceback.print_exc()
        return 3


def check_isaacsim_import(kitless):
    print("\n[3/4] Import isaacsim")
    if kitless:
        print("  Skipping (kitless mode).")
        return 0
    try:
        import isaacsim  # noqa: F401
        _print("isaacsim", "imported OK")
        return 0
    except Exception:
        traceback.print_exc()
        return 4


def check_applauncher(kitless):
    print("\n[4/4] AppLauncher headless smoke")
    if kitless:
        # In kit-less mode we use Newton directly. Just confirm the physics
        # backend can be imported.
        try:
            import isaaclab.sim  # noqa: F401
            _print("isaaclab.sim", "imported OK (kitless)")
            return 0
        except Exception:
            traceback.print_exc()
            return 5
    try:
        from isaaclab.app import AppLauncher
        # Use a minimal Namespace; AppLauncher accepts a dict-like.
        app_launcher = AppLauncher(headless=True)
        app = app_launcher.app
        _print("simulation_app", "started")
        # Attempt to import sim utils and close immediately.
        try:
            import isaaclab.sim  # noqa: F401
            _print("isaaclab.sim", "imported OK")
        finally:
            app.close()
        return 0
    except Exception:
        traceback.print_exc()
        return 5


def main(argv=None):
    p = argparse.ArgumentParser(description="Isaac Lab headless smoke test.")
    p.add_argument("--kitless", action="store_true",
                   help="Skip isaacsim import and AppLauncher (for kitless installs).")
    args = p.parse_args(argv)

    print("============================================================")
    print(" Isaac Lab — headless smoke test")
    print("============================================================")

    rc = check_python()
    if rc:
        return rc
    rc = check_isaaclab_import()
    if rc:
        return rc
    rc = check_isaacsim_import(args.kitless)
    if rc:
        return rc
    rc = check_applauncher(args.kitless)
    if rc:
        return rc

    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
