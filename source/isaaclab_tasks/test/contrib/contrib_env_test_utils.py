# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared parametrization for the contributed-environment smoke tests.

The smoke tests are split by the runtime each environment needs, so a test process only starts what its
environments use: ``kitless`` environments run without Isaac Sim, ``kit`` environments need Isaac Sim for
PhysX but no renderer, and ``kit_cameras`` environments also need the RTX renderer, the expensive part of
starting Isaac Sim. The runtime comes from :func:`isaaclab.app.sim_launcher.scan`, the same check
``launch_simulation`` uses to decide whether to start Isaac Sim, so a new environment lands in the right file
without being listed anywhere.
"""

from typing import Literal

import pytest

from isaaclab.app.sim_launcher import scan

from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

# Local imports should be imported last
from env_test_utils import setup_environment  # isort: skip

Runtime = Literal["kitless", "kit", "kit_cameras"]

_SKIPPED_TASKS = {
    "IsaacContrib-AutoMate-Assembly-Direct": "Requires CUDA support outside the standard environment test runner.",
    "IsaacContrib-AutoMate-Disassembly-Direct": "Requires CUDA support outside the standard environment test runner.",
}
_SKIPPED_TASK_SUBSTRINGS = {
    # Under random actions the Kamino P-ADMM solver intermittently diverges and the whole robot state
    # (root pose, joint state) turns NaN mid-episode, so the run fails nondeterministically (about 1 in 12
    # seeds locally; the sibling HoldPose task stays finite). The termination terms cannot catch a NaN state.
    # Re-enable once the solver instability is resolved upstream.
    "DrLegs-Walk": "Kamino solver intermittently produces NaN robot state under random actions.",
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


def task_runtime(task_name: str) -> Runtime:
    """Return the runtime an environment's default configuration launches with."""
    config_scan = scan(parse_env_cfg(task_name))
    if not config_scan.needs_kit:
        return "kitless"
    return "kit_cameras" if config_scan.has_kit_camera else "kit"


def contrib_environment_params(runtime: Runtime) -> list:
    """Return each contributed environment that launches with ``runtime``, with its documented test marks."""
    params = []
    for task_param in setup_environment(multi_agent=False, tier="contrib", exclude_task_names=_COVERED_TASKS):
        task_name = getattr(task_param, "values", (task_param,))[0]
        if task_runtime(task_name) != runtime:
            continue
        marks = getattr(task_param, "marks", ())
        if (skip_reason := _skip_reason(task_name)) is not None:
            marks = (*marks, pytest.mark.skip(reason=skip_reason))
        params.append(pytest.param(task_name, id=task_name, marks=marks))
    return params


def num_envs(task_name: str) -> int:
    """Return how many environments the smoke test steps for a task."""
    return 3 if task_name == "IsaacContrib-Multitask-Manipulation" else 2
