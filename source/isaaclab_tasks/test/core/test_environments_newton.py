# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Smoke tests for core environments using the Newton MJWarp runtime."""

import os

# TODO: Remove once usd-core>=26.5 is the minimum. Earlier releases can corrupt
# the heap while parsing Newton payloads concurrently, so disable USD concurrency
# before importing modules that may initialize OpenUSD.
os.environ["PXR_WORK_THREAD_LIMIT"] = "1"

import pytest
import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.sim import SimulationContext

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.core.velocity.config.g1.rough_env_cfg import G1RoughEnvCfg
from isaaclab_tasks.utils.hydra import resolve_presets

# Local imports should be imported last
from env_test_utils import SINGLE_ENVIRONMENT_TASKS, _run_environments, setup_environment  # isort: skip


_COVERED_TASKS = [
    "Isaac-Cartpole",  # Already covered by test_environment_determinism.py
    "Isaac-Cartpole-Camera-Direct",  # Already covered by test_rendering_cartpole.py
    "Isaac-Lift-KukaAllegro-Camera",  # Already covered by test_rendering_lift_kuka_homo_kitless.py
    "Isaac-Reorient-Cube-Shadow-Camera-Direct",  # Already covered by test_rendering_shadow_hand_kitless.py
]

_ENVIRONMENT_TASKS = setup_environment(
    multi_agent=False,
    physics_preset_name="newton_mjwarp",
    tier="core",
    exclude_task_names=_COVERED_TASKS,
)


@pytest.mark.parametrize(
    "task_name",
    _ENVIRONMENT_TASKS,
)
@pytest.mark.parametrize("num_envs, device", [(2, "cuda")])
def test_environments_newton(task_name, num_envs, device):
    _run_environments(task_name, device, num_envs, physics_preset_name="newton_mjwarp")


@pytest.mark.parametrize("task_name", [task for task in _ENVIRONMENT_TASKS if task in SINGLE_ENVIRONMENT_TASKS])
def test_single_environment_newton(task_name):
    _run_environments(task_name, "cuda", 1, physics_preset_name="newton_mjwarp")


def test_in_step_reset_updates_newton_ray_caster_pose():
    """The first observation after a timeout uses post-reset articulation kinematics."""
    env_cfg = resolve_presets(G1RoughEnvCfg(), selected=("newton_mjwarp",))
    env_cfg.scene.num_envs = 1
    env_cfg.scene.terrain.terrain_type = "plane"
    env_cfg.scene.terrain.terrain_generator = None
    env_cfg.curriculum = None
    env_cfg.terminations.base_contact = None
    env_cfg.episode_length_s = env_cfg.sim.dt * env_cfg.decimation

    env = ManagerBasedRLEnv(cfg=env_cfg)
    try:
        env.reset()
        reset_base_cfg = env.event_manager.get_term_cfg("reset_base")
        reset_base_cfg.params["pose_range"]["x"] = (1.0, 1.0)
        reset_base_cfg.params["pose_range"]["y"] = (1.0, 1.0)
        env.event_manager.set_term_cfg("reset_base", reset_base_cfg)

        action = torch.zeros((1, env.action_space.shape[-1]), device=env.device)
        with torch.inference_mode():
            _, _, _, truncated, _ = env.step(action)

        assert truncated.item()
        robot_pos = env.scene["robot"].data.root_link_pos_w.torch[0, :2]
        scanner_pos = env.scene["height_scanner"].data.pos_w.torch[0, :2]
        torch.testing.assert_close(scanner_pos, robot_pos)
    finally:
        env.close()
        SimulationContext.clear_instance()
