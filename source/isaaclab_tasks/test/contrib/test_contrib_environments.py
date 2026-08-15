# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

import sys

# Import pinocchio before AppLauncher so Isaac Lab's dependency wins over Isaac Sim's bundled copy.
if sys.platform != "win32":
    import pinocchio  # noqa: F401

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True, enable_cameras=True)
simulation_app = app_launcher.app


"""Rest everything follows."""

import pytest

import isaaclab_tasks  # noqa: F401

# Local imports should be imported last
from env_test_utils import _run_environments, setup_environment  # isort: skip


_SKIPPED_TASKS = {
    "IsaacContrib-Franka-Pour": "Requires an external reset-dataset artifact.",
    "IsaacContrib-AutoMate-Assembly-Direct": "Requires CUDA support outside the standard environment test runner.",
    "IsaacContrib-AutoMate-Disassembly-Direct": "Requires CUDA support outside the standard environment test runner.",
}
_SKIPPED_TASK_SUBSTRINGS = {
    "RmpFlow": "Uses SingleArticulation, which requires an update.",
    "Skillgen": "Requires cuRobo-specific coverage.",
    "Suction": "Requires CPU simulation.",
}
_COVERED_TASKS = [
    "IsaacContrib-Lift-Cube-Franka",  # Already covered by test_environment_determinism.py
]


def _skip_reason(task_name: str) -> str | None:
    """Return the documented reason for skipping a contributed environment."""
    if task_name in _SKIPPED_TASKS:
        return _SKIPPED_TASKS[task_name]
    return next((reason for substring, reason in _SKIPPED_TASK_SUBSTRINGS.items() if substring in task_name), None)


def _contrib_environment_params() -> list:
    """Return each contributed environment with its documented test marks."""
    params = []
    for task_param in setup_environment(
        multi_agent=False,
        tier="contrib",
        exclude_task_names=_COVERED_TASKS,
    ):
        task_name = getattr(task_param, "values", (task_param,))[0]
        marks = getattr(task_param, "marks", ())
        skip_reason = _skip_reason(task_name)
        if skip_reason is not None:
            marks = (*marks, pytest.mark.skip(reason=skip_reason))
        params.append(pytest.param(task_name, id=task_name, marks=marks))
    return params


@pytest.mark.parametrize("task_name", _contrib_environment_params())
def test_contrib_environments(task_name):
    _run_environments(task_name, device="cuda", num_envs=2)
