# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The particle-push paddle follows its wrist after cloning and joint-state writes."""

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg

from isaaclab.sim import SimulationCfg
from isaaclab.test.utils import DeviceScope, launch_test_simulation, test_devices

SIM_CFG = SimulationCfg(physics=NewtonCfg(solver_cfg=MJWarpSolverCfg(), load_visual_shapes=True))
launch_test_simulation(SIM_CFG)

import pytest
import torch
from isaaclab_newton.physics import NewtonManager
from newton import ShapeFlags

from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveScene
from isaaclab.sim import build_simulation_context
from isaaclab.utils import math as math_utils
from isaaclab.utils import replace

from isaaclab_tasks.contrib.ur10_particle_push.ur10_particle_push_env_cfg import (
    PADDLE_OFFSET,
    UR10ParticlePushSceneCfg,
)


@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
def test_paddle_follows_wrist_after_cloning_and_joint_state_writes(device):
    """The task's physical and visible tool stay on the wrist in both cloned worlds."""
    scene_cfg = UR10ParticlePushSceneCfg(num_envs=2, env_spacing=3.0)
    # Exercise the task's robot assembly without building the unrelated MPM workcell.
    for name, asset_cfg in vars(scene_cfg).items():
        if isinstance(asset_cfg, AssetBaseCfg) and not asset_cfg.prim_path.startswith(scene_cfg.robot.prim_path):
            setattr(scene_cfg, name, None)

    with build_simulation_context(sim_cfg=replace(SIM_CFG, device=device)) as sim:
        scene = InteractiveScene(scene_cfg)
        sim.reset()
        robot = scene["robot"]
        wrist_id = robot.find_bodies("ee_link")[0][0]
        model = NewtonManager.backend.model
        paddle_paths = [f"/World/envs/env_{world}/Robot/ee_link/Paddle" for world in range(2)]
        paddle_ids = [model.body_label.index(path) for path in paddle_paths]
        joint_pos = robot.data.default_joint_pos.torch.clone()
        offset = torch.tensor(PADDLE_OFFSET, device=device).expand(2, -1)
        for displacement in (0.0, 0.3):
            joint_pos[:, 0] += displacement
            robot.write_joint_state_to_sim_index(position=joint_pos, velocity=torch.zeros_like(joint_pos))
            wrist_pose = robot.data.body_link_pose_w.torch[:, wrist_id]
            expected_pos = wrist_pose[:, :3] + math_utils.quat_apply(wrist_pose[:, 3:], offset)
            actual_pose = torch.as_tensor(NewtonManager.backend.state_0.body_q.numpy(), device=device)[paddle_ids]
            torch.testing.assert_close(actual_pose[:, :3], expected_pos, atol=1.0e-5, rtol=0.0)
            rotation_error = math_utils.quat_error_magnitude(actual_pose[:, 3:], wrist_pose[:, 3:])
            torch.testing.assert_close(rotation_error, torch.zeros_like(rotation_error), atol=1.0e-5, rtol=0.0)

        for path, body_id in zip(paddle_paths, paddle_ids, strict=True):
            visual_id = model.shape_label.index(path + "/PaddleVisual/geometry/mesh")
            assert model.shape_body.numpy()[visual_id] == body_id
            assert model.shape_flags.numpy()[visual_id] & ShapeFlags.VISIBLE
