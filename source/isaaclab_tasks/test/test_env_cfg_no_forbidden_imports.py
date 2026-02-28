# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Test that env_cfg construction does not import forbidden backend modules.

``load_cfg_from_registry`` runs BEFORE SimulationApp is launched to inspect
the physics backend and decide whether Kit is needed at all.  Config classes
are pure data — they must be constructable without any runtime dependencies
that conflict with Kit's internal ``fork()`` or that require a running
simulator.

Forbidden categories
--------------------
1. **Backend / simulator runtime** (``pxr``, ``omni``, ``carb``, ``isaacsim``)
   — require SimulationApp / Kit to be initialized first.
2. **SciPy** — loads OpenBLAS which registers ``atfork`` handlers that crash
   Kit's internal ``fork()`` during startup.

Remediation patterns
--------------------
* Use ``lazy_loader.attach_stub`` in ``__init__.py`` files with a
  corresponding ``.pyi`` stub so that implementation modules are only
  imported when first accessed.
* Guard annotation-only imports with ``TYPE_CHECKING``.
* Store ``class_type`` / ``func`` fields as fully-qualified strings
  (e.g. ``"isaaclab.assets.articulation:Articulation"``); ``cfg.validate()``
  resolves them to callables after Kit has launched.
* Use local ``# noqa: PLC0415`` imports inside functions for Kit-dependent
  symbols that cannot be imported at module level before Kit is running.

Performance note
----------------
All task checks are batched into a **single subprocess** so that
``import isaaclab_tasks`` (~1.6 s) is paid only once instead of once per test.
Results are returned as JSON and cached for the parametrized test functions.
"""

import json
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
    "pxr",  # USD Python bindings
    "omni",  # Omniverse runtime
    "carb",  # Carbonite framework
    "isaacsim",  # Isaac Sim modules
    # SciPy loads OpenBLAS which crashes Kit's fork()
    "scipy",
)

_ALL_ISAAC_TASKS = sorted(name for name in gymnasium.registry if name.startswith("Isaac-"))

# ---------------------------------------------------------------------------
# Batch subprocess: run all checks in one Python process so we only pay the
# `import isaaclab_tasks` cost once (~1.6 s) instead of once per test.
# ---------------------------------------------------------------------------


def _build_batch_script(task_names: list[str]) -> str:
    return textwrap.dedent(f"""\
        import sys, traceback, json

        FORBIDDEN = {list(_FORBIDDEN_PREFIXES)!r}
        task_names = {task_names!r}

        import isaaclab_tasks  # noqa: F401
        from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

        results = {{}}

        for task_name in task_names:
            violations = {{}}
            load_error = None

            _orig_import = __builtins__.__import__

            def _hook(name, *args, **kw):
                top = name.split('.')[0]
                if top in FORBIDDEN and top not in violations:
                    violations[top] = ''.join(traceback.format_stack())
                return _orig_import(name, *args, **kw)

            __builtins__.__import__ = _hook
            try:
                cfg = load_cfg_from_registry(task_name, "env_cfg_entry_point")
            except Exception as exc:
                load_error = str(exc)
            finally:
                __builtins__.__import__ = _orig_import

            # Note: we intentionally do NOT clean up sys.modules between tasks.
            # The import hook intercepts every __import__ call regardless of
            # whether the module is already cached, so it reliably catches
            # violations even across tasks.  Cleaning up sys.modules would force
            # re-importing shared modules (e.g. velocity_env_cfg, which is
            # shared by 40+ locomotion tasks) and turn a 3 s run into 17 s.

            results[task_name] = {{
                'load_error': load_error,
                'violations': violations,
            }}

        # Use a sentinel so the parser can find the JSON even when
        # load_cfg_from_registry prints [INFO] lines to stdout.
        print("__RESULTS__" + json.dumps(results))
    """)


@pytest.fixture(scope="session")
def all_cfg_check_results() -> dict:
    """Run all task cfg checks in a single subprocess and return results dict."""
    script = _build_batch_script(_ALL_ISAAC_TASKS)
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=300,
    )
    # Find the sentinel line (load_cfg_from_registry emits [INFO] lines to stdout)
    json_line = None
    for line in result.stdout.splitlines():
        if line.startswith("__RESULTS__"):
            json_line = line[len("__RESULTS__") :]
            break

    if json_line is None:
        return {
            "__subprocess_crash__": (
                f"Batch subprocess did not produce results.\n"
                f"--- stdout ---\n{result.stdout}\n"
                f"--- stderr ---\n{result.stderr}"
            )
        }
    try:
        return json.loads(json_line)
    except json.JSONDecodeError as exc:
        return {"__json_error__": str(exc), "__raw__": json_line[:500]}


@pytest.mark.parametrize("task_name", _ALL_ISAAC_TASKS)
def test_config_load_does_not_import_backend_modules(task_name: str, all_cfg_check_results: dict):
    """Config loading must not import forbidden runtime modules.

    Config classes are pure data.  They must not pull in backend modules
    (pxr, omni, carb, isaacsim) or heavyweight libraries (scipy).

    Fix: use lazy_loader.attach_stub with .pyi stubs in __init__.py files,
    TYPE_CHECKING guards for annotation-only imports, and string references
    for class_type/func fields in cfg files.
    """
    if "__subprocess_crash__" in all_cfg_check_results:
        pytest.fail(f"Batch check subprocess crashed:\n{all_cfg_check_results['__subprocess_crash__']}")

    if task_name not in all_cfg_check_results:
        pytest.fail(f"No result for '{task_name}' - batch subprocess may have crashed.")

    info = all_cfg_check_results[task_name]
    load_error = info.get("load_error")
    violations = info.get("violations", {})

    messages = []
    if load_error:
        messages.append(f"ERROR: config load crashed: {load_error}")
    if violations:
        messages.append(f"FAIL: {len(violations)} forbidden top-level module(s) imported:")
        for mod, stack in sorted(violations.items()):
            messages.append(f"\n=== {mod} ===\n{stack}")

    assert not violations and not load_error, (
        f"Config loading for '{task_name}' imported forbidden backend modules.\n"
        f"Forbidden prefixes: {_FORBIDDEN_PREFIXES}\n"
        + "\n".join(messages)
        + "\n\nFix: use lazy_loader.attach_stub with a .pyi stub in the offending "
        "__init__.py, or move the import under TYPE_CHECKING and use a string "
        "reference for isinstance checks."
    )
