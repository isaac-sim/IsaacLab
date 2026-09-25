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

Many contributed environments are variants of one another: the same task and robot with other colors, action
spaces, or an inference wrapper. A smoke test catches an environment that no longer builds, steps, or resets,
which variants share, so only one environment per task package, robot, and runtime is smoke-tested.
"""

from collections import defaultdict
from typing import Literal

import gymnasium as gym
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


def _variant_family(task_name: str) -> tuple[str, str]:
    """Return the task package and robot directory an environment's configuration is defined in.

    Contributed tasks keep per-robot configurations under ``<task package>/config/<robot>/``; a task without
    that layout is its own robot.
    """
    entry_point = gym.spec(task_name).kwargs["env_cfg_entry_point"]
    module = entry_point.partition(":")[0] if isinstance(entry_point, str) else entry_point.__module__
    parts = module.split(".")
    if "config" not in parts[:-1]:
        return ".".join(parts[:-1]), ""
    config_index = parts.index("config")
    robot = parts[config_index + 1] if config_index + 1 < len(parts) - 1 else ""
    return ".".join(parts[:config_index]), robot


def contrib_environment_params(runtime: Runtime) -> list:
    """Return one contributed environment per task package, robot, and runtime, for the tasks that use ``runtime``.

    Each family is represented by an environment that is not skipped, preferring the shortest task ID (usually
    the base variant). A family whose environments are all skipped keeps one, so its skip reason stays visible.
    """
    tasks_by_family: dict[tuple[str, str, Runtime], list[str]] = defaultdict(list)
    task_marks = {}
    for task_param in setup_environment(multi_agent=False, tier="contrib", exclude_task_names=_COVERED_TASKS):
        task_name = getattr(task_param, "values", (task_param,))[0]
        task_marks[task_name] = getattr(task_param, "marks", ())
        tasks_by_family[(*_variant_family(task_name), task_runtime(task_name))].append(task_name)

    params = []
    for (_, _, family_runtime), task_names in sorted(tasks_by_family.items()):
        if family_runtime != runtime:
            continue
        task_name = min(task_names, key=lambda name: (_skip_reason(name) is not None, len(name), name))
        marks = task_marks[task_name]
        if (skip_reason := _skip_reason(task_name)) is not None:
            marks = (*marks, pytest.mark.skip(reason=skip_reason))
        params.append(pytest.param(task_name, id=task_name, marks=marks))
    return params


def num_envs(task_name: str) -> int:
    """Return how many environments the smoke test steps for a task."""
    return 3 if task_name == "IsaacContrib-Multitask-Manipulation" else 2
