# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Registered camera tasks retain their authored stage structure independently of pixel tests."""

from isaaclab.test.utils import launch_test_simulation

launch_test_simulation(enable_cameras=True)

from pathlib import Path

import gymnasium as gym
import pytest
from stage_comparison import compare_golden_stage

from isaaclab.sim import save_stage

from isaaclab_tasks.utils.hydra import resolve_presets
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

pytestmark = pytest.mark.isaacsim_ci


@pytest.mark.parametrize("task", ("Isaac-Cartpole-Camera-Direct", "Isaac-Reorient-Cube-Shadow-Camera-Direct"))
def test_registered_task_stage(task, tmp_path):
    cfg = resolve_presets(
        load_cfg_from_registry(task, "env_cfg_entry_point"), {"isaacsim_physx", "isaacsim_rtx", "rgb"}
    )
    cfg.scene.num_envs = 4
    cfg.sim.device = "cuda:0"
    env = gym.make(task, cfg=cfg)
    try:
        result = tmp_path / "stage.usda"
        assert save_stage(str(result), save_and_reload_in_place=False)
        golden = (
            Path(__file__).parents[1]
            / "golden_stages"
            / f"registered_tasks_{task}/default_physics-default_renderer-stage.usda"
        )
        assert golden.is_file()
        assert not (problems := compare_golden_stage(str(golden), str(result))), "\n".join(problems)
    finally:
        env.close()
