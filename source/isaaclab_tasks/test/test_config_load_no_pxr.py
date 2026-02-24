# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Test that config loading does not import modules that crash Kit's startup.

Config loading (``load_cfg_from_registry``) runs BEFORE SimulationApp to
inspect the physics backend and decide whether Kit is needed at all.
Config classes are pure data — they must be constructable without any
runtime dependencies that conflict with Kit's internal ``fork()``.

Forbidden categories:
  1. Backend / simulator modules (pxr, omni, carb, isaacsim) — require
     SimulationApp / Kit to be initialized first.
  2. SciPy — loads OpenBLAS which registers atfork handlers that crash
     Kit's internal fork() during startup.

The only allowed mechanism for deferring heavy imports is ``lazy_loader``
(``lazy.attach``).  No other lazy-load hacks (manual ``__getattr__``,
wrapper callables, etc.) should be introduced.
"""

import subprocess
import sys
import textwrap

import gymnasium
import pytest

import isaaclab_tasks  # noqa: F401 -- triggers task registration

# Forbidden module prefixes -- these must NOT appear in sys.modules after
# config loading because they require SimulationApp / a specific physics
# backend to be started first, or because they are heavyweight runtime
# libraries that should never be needed to construct pure-data config objects.
_FORBIDDEN_PREFIXES = (
    # Backend / simulator runtime (require SimulationApp / Kit)
    "pxr",        # USD Python bindings
    "omni",       # Omniverse runtime
    "carb",       # Carbonite framework
    "isaacsim",   # Isaac Sim modules
    # SciPy loads OpenBLAS which crashes Kit's fork()
    "scipy",
)

_ALL_ISAAC_TASKS = sorted(name for name in gymnasium.registry if name.startswith("Isaac-"))


def _build_check_script(task_name: str) -> str:
    """Return a self-contained Python script that loads a task config and
    checks for forbidden imports.

    Uses ``__builtins__.__import__`` hook which intercepts every single
    import at the lowest level — more reliable than ``sys.meta_path``
    finders which can be bypassed by C extensions, cached modules, or
    custom loaders.

    The hook does NOT block or stub imports.  It lets the real import
    proceed and simply records the first occurrence of each forbidden
    top-level module together with the stack trace that triggered it.
    This way the full import chain is preserved for diagnostics without
    altering runtime behaviour.
    """
    return textwrap.dedent(f"""\
        import sys
        import traceback

        _FORBIDDEN = {_FORBIDDEN_PREFIXES!r}
        _violations = {{}}          # top_module -> stack trace (first occurrence only)
        _original_import = __builtins__.__import__

        def _tracing_import(name, *args, **kwargs):
            top = name.split(".")[0]
            if top in _FORBIDDEN and top not in _violations:
                _violations[top] = traceback.format_stack()
            return _original_import(name, *args, **kwargs)

        __builtins__.__import__ = _tracing_import

        # ---- load the task config ----
        _load_error = None
        try:
            import isaaclab_tasks  # noqa: F401
            from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
            cfg = load_cfg_from_registry("{task_name}", "env_cfg_entry_point")
        except Exception as exc:
            _load_error = str(exc)

        # ---- report ----
        if _load_error:
            print(f"ERROR: config load crashed: {{_load_error}}", file=sys.stderr)

        if _violations:
            print(f"FAIL: {{len(_violations)}} forbidden top-level module(s) imported:", file=sys.stderr)
            for top_mod, stack_frames in sorted(_violations.items()):
                print(f"\\n=== {{top_mod}} ===", file=sys.stderr)
                # show only the interesting frames (skip importlib internals)
                for frame in stack_frames:
                    if "importlib" not in frame and "<frozen" not in frame:
                        print(frame.rstrip(), file=sys.stderr)
            sys.exit(1)

        # also check sys.modules as a safety net
        leaked = sorted(
            m for m in sys.modules
            if any(m == p or m.startswith(p + ".") for p in _FORBIDDEN)
        )
        if leaked:
            print(f"FAIL: forbidden modules found in sys.modules: {{leaked}}", file=sys.stderr)
            sys.exit(1)

        print("PASS")
    """)


def _run_config_load_check(task_name: str) -> tuple[int, str, str]:
    """Run the check in a subprocess so the current process stays clean."""
    result = subprocess.run(
        [sys.executable, "-c", _build_check_script(task_name)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    return result.returncode, result.stdout, result.stderr


@pytest.mark.parametrize("task_name", _ALL_ISAAC_TASKS)
def test_config_load_does_not_import_backend_modules(task_name: str):
    """Config loading must not import forbidden runtime modules.

    Config classes are pure data.  They must not pull in backend modules
    (pxr, omni, carb, isaacsim), heavyweight compute libraries (torch,
    scipy, numpy, warp), or I/O libraries (h5py, trimesh, PIL, lxml, rtree).

    Fix: use lazy_loader (lazy.attach) in __init__.py files, TYPE_CHECKING
    guards, and string references / DeferredClass for implementation classes.
    """
    rc, stdout, stderr = _run_config_load_check(task_name)
    assert rc == 0, (
        f"Config loading for '{task_name}' imported forbidden backend modules.\n"
        f"Forbidden prefixes: {_FORBIDDEN_PREFIXES}\n"
        f"The import chain(s) for each violation are shown below.\n"
        f"--- stderr ---\n{stderr}\n--- stdout ---\n{stdout}\n"
        "Fix: use lazy_loader (lazy.attach) in the offending __init__.py, "
        "or move the import under TYPE_CHECKING and use a string reference / "
        "_is_instance_by_name() for isinstance checks."
    )
