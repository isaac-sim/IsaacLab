# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Test that the Hydra-based resolve_task_config flow for factory_v1 tasks
does not import forbidden backend modules before SimulationApp.

This complements test_env_cfg_no_forbidden_imports.py (which tests
load_cfg_from_registry alone) by exercising the full resolve_task_config →
Hydra → register_task pipeline that the actual training script uses.
"""

import json
import subprocess
import sys
import textwrap

import pytest

_FORBIDDEN_PREFIXES = ("pxr", "omni", "carb", "isaacsim", "scipy")

_FACTORY_TASKS = [
    "IsaacContrib-Factory-Franka",
]


def _build_hydra_check_script(task_name: str) -> str:
    return textwrap.dedent(f"""\
        import sys, traceback, json

        FORBIDDEN = {list(_FORBIDDEN_PREFIXES)!r}
        task_name = {task_name!r}

        violations = {{}}
        load_error = None

        import isaaclab_tasks  # noqa: F401

        _orig_import = __builtins__.__import__

        def _hook(name, *args, **kw):
            top = name.split('.')[0]
            if top in FORBIDDEN and top not in violations:
                violations[top] = ''.join(traceback.format_stack())
            return _orig_import(name, *args, **kw)

        __builtins__.__import__ = _hook
        try:
            from isaaclab_tasks.utils.hydra import resolve_task_config
            env_cfg, agent_cfg = resolve_task_config(task_name, "rsl_rl_cfg_entry_point")
        except Exception as exc:
            load_error = str(exc)
        finally:
            __builtins__.__import__ = _orig_import

        result = {{
            'load_error': load_error,
            'violations': violations,
        }}
        print("__RESULTS__" + json.dumps(result))
    """)


@pytest.mark.parametrize("task_name", _FACTORY_TASKS)
def test_hydra_resolve_does_not_import_backend_modules(task_name: str):
    """resolve_task_config (Hydra flow) must not import forbidden modules.

    The training script calls resolve_task_config *before* SimulationApp is
    launched. Any pxr/omni/carb/isaacsim/scipy import at that stage will
    crash Kit.
    """
    script = _build_hydra_check_script(task_name)
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=120,
    )

    json_line = None
    for line in result.stdout.splitlines():
        if line.startswith("__RESULTS__"):
            json_line = line[len("__RESULTS__") :]
            break

    if json_line is None:
        pytest.fail(
            f"Subprocess did not produce results for '{task_name}'.\n"
            f"--- stdout ---\n{result.stdout}\n"
            f"--- stderr ---\n{result.stderr}"
        )

    info = json.loads(json_line)
    load_error = info.get("load_error")
    violations = info.get("violations", {})

    messages = []
    if load_error:
        messages.append(f"ERROR: resolve_task_config crashed: {load_error}")
    if violations:
        messages.append(f"FAIL: {len(violations)} forbidden top-level module(s) imported:")
        for mod, stack in sorted(violations.items()):
            messages.append(f"\n=== {mod} ===\n{stack}")

    assert not violations and not load_error, (
        f"resolve_task_config for '{task_name}' imported forbidden backend modules.\n"
        f"Forbidden prefixes: {_FORBIDDEN_PREFIXES}\n" + "\n".join(messages)
    )
