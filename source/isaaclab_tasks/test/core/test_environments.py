# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch the simulator
app_launcher = AppLauncher(headless=True, enable_cameras=True, limit_cpu_threads=1)
simulation_app = app_launcher.app


"""Rest everything follows."""

import subprocess
import sys
from pathlib import Path

import pytest

import isaaclab_tasks  # noqa: F401

# Local imports should be imported last
from env_test_utils import _run_environments, setup_environment  # isort: skip


def _ensure_franka_pour_reset_dataset() -> None:
    """Generate the smallest valid reset dataset when the Pour smoke test needs it."""
    repo_root = Path(__file__).resolve().parents[4]
    dataset_path = repo_root / "datasets/franka_pour/reset_dataset.pt"
    if dataset_path.is_file():
        return

    subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts/tools/generate_franka_pour_reset_dataset.py"),
            "--device",
            "cuda:0",
            "--grasping_count",
            "100",
            "--non_grasping_count",
            "6",
            "--batch_size",
            "128",
        ],
        cwd=repo_root,
        check=True,
    )


@pytest.mark.parametrize("physics_preset_name", ["newton_mjwarp", "physx", "isaacsim_physx"])
@pytest.mark.parametrize("num_envs, device", [(2, "cuda"), (1, "cuda")])
@pytest.mark.parametrize(
    "task_name",
    setup_environment(
        multi_agent=False,
        tier="core",
    ),
)
@pytest.mark.isaacsim_ci
def test_environments(task_name, physics_preset_name, num_envs, device):
    # run environments without stage in memory
    _run_environments(
        task_name, device, num_envs, create_stage_in_memory=False, physics_preset_name=physics_preset_name
    )


@pytest.mark.parametrize("num_envs, device", [(2, "cuda")])
@pytest.mark.parametrize(
    "task_name",
    setup_environment(
        multi_agent=False,
        factory_envs=False,
        cartpole_showcase_envs=False,
        pickplace_stack_envs=False,
        teleop_envs=False,
        tier="contrib",
    ),
)
@pytest.mark.isaacsim_ci
def test_contrib_environments(task_name, num_envs, device):
    if task_name == "IsaacContrib-Franka-Pour":
        _ensure_franka_pour_reset_dataset()
    _run_environments(task_name, device, num_envs, create_stage_in_memory=False)
