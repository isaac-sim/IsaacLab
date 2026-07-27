# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""GPU-free unit tests for backend identity normalization."""

from __future__ import annotations

import sys
from pathlib import Path

_GATE_DIR = Path(__file__).resolve().parents[1]
if str(_GATE_DIR) not in sys.path:
    sys.path.insert(0, str(_GATE_DIR))

from backend_identity import identity_from_presets  # noqa: E402
from launch_config import hydra_args_for_task  # noqa: E402
from task_config import load_tasks  # noqa: E402


def test_identity_from_presets_treats_newton_as_newton() -> None:
    """The 6.0 prerelease task presets report ``newton`` instead of ``newton_mjwarp``."""
    identity = identity_from_presets("newton")

    assert identity is not None
    assert identity.backend_key == "newton"


def test_identity_from_presets_keeps_newton_mjwarp_compatibility() -> None:
    """The newer preset spelling remains supported."""
    identity = identity_from_presets("cube,newton_mjwarp")

    assert identity is not None
    assert identity.backend_key == "newton"


def test_camera_tasks_explicitly_identify_their_rtx_renderer() -> None:
    """Camera buckets match the RTX renderer that the task activates by default."""
    camera_task = "Isaac-Reorient-Cube-Shadow-Camera-Benchmark-Direct"
    backend_keys = {task.backend_key for task in load_tasks() if task.task_id == camera_task}

    assert backend_keys == {
        "physx_rtx_renderer",
        "physx_newton_renderer",
        "newton_rtx_renderer",
        "newton_newton_renderer",
    }


def test_rtx_camera_tasks_use_the_supported_isaac_sim_preset() -> None:
    """Canonical RTX identity maps to the Hydra preset exposed by Isaac Lab."""
    camera_task = "Isaac-Reorient-Cube-Shadow-Camera-Benchmark-Direct"
    rtx_tasks = [task for task in load_tasks() if task.task_id == camera_task and task.render_backend == "rtx_renderer"]

    assert {tuple(hydra_args_for_task(task)) for task in rtx_tasks} == {
        ("presets=isaacsim_rtx",),
        ("presets=newton_mjwarp,isaacsim_rtx",),
    }
