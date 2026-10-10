# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Position commands stay rate bounded across saturation, reversal, and partial resets."""

from isaaclab_newton.physics import NewtonCfg

from isaaclab.sim import SimulationCfg
from isaaclab.test.utils import launch_test_simulation

launch_test_simulation(SimulationCfg(physics=NewtonCfg()))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402

from isaaclab_tasks.core.cabinet.config.franka.joint_pos_env_cfg import FrankaCabinetEnvCfg  # noqa: E402
from isaaclab_tasks.utils.hydra import resolve_presets  # noqa: E402


def test_cabinet_reset_clearance_and_bounded_targets():
    cfg = resolve_presets(FrankaCabinetEnvCfg(), {"newton_mjwarp"})
    cfg.scene.num_envs = 2
    env = ManagerBasedRLEnv(cfg)
    try:
        env.reset()
        robot = env.scene["robot"]
        # Check the reset distribution against the actual imported collision geometry.
        model = env.sim.physics_manager._solver.mj_model
        data = mujoco.MjData(model)
        addresses = []
        for name in robot.joint_names:
            joint_id = next(
                i
                for i in range(model.njnt)
                if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i).endswith("_" + name)
            )
            addresses.append(model.jnt_qposadr[joint_id])
        default = robot.data.default_joint_pos.torch[0].cpu().numpy()
        reset_limits = robot.data.soft_joint_pos_limits.torch[0].cpu().numpy()
        rng = np.random.default_rng(42)
        low, high = cfg.events.reset_robot_joints.params["position_range"]
        samples = np.clip(
            default + rng.uniform(low, high, (4096, len(default))), reset_limits[:, 0], reset_limits[:, 1]
        )
        for pose in samples:
            data.qpos[addresses] = pose
            mujoco.mj_forward(model, data)
            assert all(contact.dist >= -1e-4 for contact in data.contact), "Reset pose penetrates the scene"

        joint_ids = robot.find_joints("panda_joint.*")[0]
        previous = robot.data.joint_pos.torch[:, joint_ids].clone()
        limits = robot.data.soft_joint_pos_limits.torch[:, joint_ids]
        action = torch.full((2, env.action_manager.total_action_dim), 1000.0, device=env.device)
        max_delta = cfg.actions.arm_action.max_velocity * env.step_dt

        # Exercise the action-manager boundary without making the physics solver part of this contract.
        for _ in range(1200):
            env.action_manager.process_action(action)
            env.action_manager.apply_action()
            target = robot.actuators.target_command.position.torch[:, joint_ids].clone()
            assert torch.all((target - previous).abs() <= max_delta + 1e-6)
            previous = target
        torch.testing.assert_close(target, limits[..., 1])

        # Only the reset environment must discard its accumulated target.
        env.reset(env_ids=torch.tensor([0], dtype=torch.int32, device=env.device))
        previous[0] = robot.data.joint_pos.torch[0, joint_ids]
        action.fill_(-1000.0)
        env.action_manager.process_action(action)
        env.action_manager.apply_action()
        target = robot.actuators.target_command.position.torch[:, joint_ids]
        torch.testing.assert_close(target, previous - max_delta)

        env.reset()
        reset_pose = robot.data.joint_pos.torch[:, joint_ids].clone()
        env.action_manager.process_action(action)
        env.action_manager.apply_action()
        torch.testing.assert_close(robot.actuators.target_command.position.torch[:, joint_ids], reset_pose - max_delta)
    finally:
        env.close()
