# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Batched coffee dynamics, tool retention, contact release and partial reset."""

import pytest
import torch
from isaaclab_newton.physics import NewtonMPMManager

from isaaclab.app import launch_simulation
from isaaclab.envs import mdp
from isaaclab.managers import EventTermCfg, SceneEntityCfg
from isaaclab.test.utils import DeviceScope, test_devices
from isaaclab.utils.math import (
    apply_delta_pose,
    combine_frame_transforms,
    quat_apply,
    quat_apply_inverse,
    subtract_frame_transforms,
)

from isaaclab_tasks.contrib.rizon_sharpa_coffee.assets import SPOON_BOWL_DEPTH, SPOON_BOWL_LENGTH, SPOON_BOWL_WIDTH
from isaaclab_tasks.contrib.rizon_sharpa_coffee.demo_env_cfg import CoffeeDemoEnvCfg
from isaaclab_tasks.contrib.rizon_sharpa_coffee.grasp_reset import configured_hand_targets, default_grasp_reset
from isaaclab_tasks.contrib.rizon_sharpa_coffee.robot import ARM_JOINT_NAMES, HAND_JOINT_NAMES, configured_asset_path


def _startup_joint_offset(env, env_ids, asset_cfg):
    """Adapt the native joint-reset term's selector to the global startup event."""
    mdp.reset_joints_by_offset(
        env,
        slice(None) if env_ids is None else env_ids,
        position_range=(0.001, 0.001),
        velocity_range=(0.0, 0.0),
        asset_cfg=asset_cfg,
    )


@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
def test_batched_dynamics_attachment_and_partial_reset(device, tmp_path):
    """Robot and coffee advance; the measured spoon follows, and reset stays local."""
    if not configured_asset_path(verify=False).is_file():
        pytest.skip("Requires the workstation's licensed RJ45 Rizon/Sharpa USD bundle.")
    cfg = CoffeeDemoEnvCfg()
    cfg.seed = 7
    cfg.scene.num_envs = 2
    cfg.voxel_size = 0.005
    cfg.sim.device = device
    cfg.sim.dt = 0.005
    cfg.decimation = 2
    cfg.sim.physics.solver_cfg.entries[0].substeps = 4
    cfg.grains_per_particle = 0
    reset = default_grasp_reset()
    reset.joint_targets["right_thumb_IP"] += 0.02
    reset_path = tmp_path / "grasp_reset.json"
    reset.save(reset_path)
    cfg.grasp_reset_path = str(reset_path)
    cfg.events.startup = EventTermCfg(
        func=_startup_joint_offset,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=list(ARM_JOINT_NAMES)),
        },
    )
    with launch_simulation(cfg, {"headless": True, "visualizer": []}):
        from isaaclab_tasks.contrib.rizon_sharpa_coffee.demo_env import CoffeeDemoEnv

        env = CoffeeDemoEnv(cfg)
        try:
            robot = env.scene["robot"]
            arm_ids, _ = robot.find_joints(list(ARM_JOINT_NAMES), preserve_order=True)
            # Configured startup events must change real state before the first reset.
            torch.testing.assert_close(
                robot.data.joint_pos.torch[:, arm_ids] - robot.data.default_joint_pos.torch[:, arm_ids],
                torch.full((2, len(arm_ids)), 0.001, device=device),
                rtol=0,
                atol=1.0e-6,
            )
            env.reset()
            robot, coffee, spoon = env.scene["robot"], env.scene["coffee"], env.scene["spoon"]
            thumb_id = robot.find_joints("right_thumb_IP")[0][0]
            torch.testing.assert_close(
                robot.actuators.target_command.position.torch[:, thumb_id] - robot.data.joint_pos.torch[:, thumb_id],
                torch.full((2,), 0.02, device=device),
                rtol=0,
                atol=1.0e-6,
            )
            initial_particles = coffee.data.particle_state_w.torch.clone()
            initial_palm = env.palm_pose().clone()
            initial_spoon = spoon.data.root_link_pose_w.torch.clone()
            hand_ids, _ = robot.find_joints(list(HAND_JOINT_NAMES), preserve_order=True)
            delta = torch.tensor((0.0, 0.0, 0.01, 0.015, -0.01, 0.008), device=device).expand(2, -1)
            command_pos, command_quat = apply_delta_pose(initial_palm[:, :3], initial_palm[:, 3:], delta)
            command = torch.cat((command_pos, command_quat), dim=-1)
            action = torch.cat((command, robot.data.default_joint_pos.torch[:, hand_ids]), dim=-1)
            for _ in range(8):
                env.step(action)
            particles = coffee.data.particle_state_w.torch.clone()
            assert torch.isfinite(particles).all()
            assert ((particles - initial_particles).abs().amax(dim=(1, 2)) > 1.0e-5).all()
            assert ((env.palm_pose()[:, 2] - initial_palm[:, 2]) > 1.0e-4).all()
            spoon_pose = spoon.data.root_link_pose_w.torch
            assert ((spoon_pose[:, :3] - initial_spoon[:, :3]).norm(dim=-1) > 1.0e-4).all()
            palm_world = robot.data.body_link_pose_w.torch[:, robot.find_bodies("r_palm_ctrl")[0][0]]
            desired_pos, desired_quat = combine_frame_transforms(
                palm_world[:, :3],
                palm_world[:, 3:],
                torch.tensor(cfg.spoon_offset_pos, device=device).expand(2, -1),
                torch.tensor(cfg.spoon_offset_rot, device=device).expand(2, -1),
            )
            # The attachment uses the measured pose at the start of the final 1.25 ms tick.
            assert ((spoon_pose[:, :3] - desired_pos).norm(dim=-1) < 0.003).all()
            assert (torch.abs((spoon_pose[:, 3:] * desired_quat).sum(dim=-1)) > 0.999).all()

            env.action_manager.apply_action()
            preserved_targets = robot.actuators.target_command.position.torch[1].clone()
            env.reset(env_ids=torch.tensor([0], device=device))
            assert robot.actuators.target_command.position.torch[0, thumb_id].item() == pytest.approx(
                reset.joint_targets["right_thumb_IP"], abs=1.0e-6
            )
            torch.testing.assert_close(coffee.data.particle_state_w.torch[0], initial_particles[0], rtol=0, atol=0)
            torch.testing.assert_close(coffee.data.particle_state_w.torch[1], particles[1], rtol=0, atol=0)
            env.action_manager.apply_action()
            torch.testing.assert_close(robot.actuators.target_command.position.torch[1], preserved_targets)
            env.step(action)
            assert torch.isfinite(coffee.data.particle_state_w.torch).all()
            assert (coffee.data.particle_state_w.torch[0] - initial_particles[0]).abs().max() > 1.0e-5

            # A valid but unreachable pose must not drive beyond joint limits.
            # Exercise the real controller and articulation command buffer in
            # this existing scene; there is no need to move the robot there.
            unreachable = action.clone()
            unreachable[:, 0] += 2.0
            env.action_manager.process_action(unreachable)
            env.action_manager.apply_action()
            arm_ids, _ = robot.find_joints(list(ARM_JOINT_NAMES), preserve_order=True)
            targets = robot.actuators.target_command.position.torch[:, arm_ids]
            limits = robot.data.soft_joint_pos_limits.torch[:, arm_ids]
            assert torch.isfinite(targets).all()
            assert (targets >= limits[..., 0]).all()
            assert (targets <= limits[..., 1]).all()

            # Fast input must remain bounded during graph replay at the coarse
            # material timestep: a 5 mm voxel permits 2.5 mm of travel per tick.
            velocity = torch.zeros_like(coffee.data.particle_vel_w.torch)
            velocity[..., 2] = 4.0
            coffee.write_particle_velocity_to_sim_index(velocity)
            env.step(action)
            assert coffee.data.particle_vel_w.torch.norm(dim=-1).max() <= 0.5001

            invalid = action.clone()
            invalid[:, 0] = torch.nan
            with pytest.raises(ValueError, match="Palm IK targets must be finite"):
                env.action_manager.process_action(invalid)
        finally:
            env.close()


@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
def test_contact_grasp_holds_releases_and_resets_locally(device):
    """A contact-held spoon retains coffee; opening one hand releases only its tool."""
    if not configured_asset_path(verify=False).is_file():
        pytest.skip("Requires the workstation's licensed RJ45 Rizon/Sharpa USD bundle.")
    cfg = CoffeeDemoEnvCfg()
    cfg.seed = 7
    cfg.scene.num_envs = 2
    cfg.voxel_size = 0.002
    cfg.coffee_fill_height = 0.009
    cfg.grasp_mode = "contact"
    cfg.sim.device = device
    cfg.sim.physics.load_visual_shapes = False
    with launch_simulation(cfg, {"headless": True, "visualizer": []}):
        from isaaclab_tasks.contrib.rizon_sharpa_coffee.demo_env import CoffeeDemoEnv

        env = CoffeeDemoEnv(cfg)
        try:
            env.reset()
            robot, spoon = env.scene["robot"], env.scene["spoon"]
            palm_id = robot.find_bodies("r_palm_ctrl")[0][0]
            initial_spoon = spoon.data.root_link_pose_w.torch.clone()
            initial_palm = robot.data.body_link_pose_w.torch[:, palm_id].clone()
            initial_offset, _ = subtract_frame_transforms(
                initial_palm[:, :3], initial_palm[:, 3:], initial_spoon[:, :3], initial_spoon[:, 3:]
            )
            # Load both spoons through the public particle-state API. A light
            # free-body MPM proxy can leak through the floor even while rigid
            # finger contacts keep the actual spoon stationary.
            local = torch.stack(
                torch.meshgrid(
                    torch.arange(-0.022, 0.023, 0.001, device=device),
                    torch.arange(-0.014, 0.015, 0.001, device=device),
                    torch.arange(0.0042, 0.013, 0.001, device=device),
                    indexing="ij",
                ),
                dim=-1,
            ).reshape(-1, 3)
            radius_squared = (local[:, 0] / (SPOON_BOWL_LENGTH / 2.0)).square() + (
                local[:, 1] / (SPOON_BOWL_WIDTH / 2.0)
            ).square()
            local = local[(radius_squared < 0.8) & (local[:, 2] > SPOON_BOWL_DEPTH * radius_squared + 0.0038)]
            count = len(local)
            coffee = env.scene["coffee"]
            state = coffee.data.particle_state_w.torch.clone()
            state[:, :count, :3] = initial_spoon[:, None, :3] + quat_apply(
                initial_spoon[:, None, 3:].expand(-1, count, -1), local.expand(2, -1, -1)
            )
            state[:, :count, 3:] = 0.0
            coffee.write_particle_state_to_sim_index(state)
            NewtonMPMManager.reset_solver_state(flags=0)
            action = torch.cat(
                (env.palm_pose(), torch.tensor(configured_hand_targets(cfg), device=device).expand(2, -1)), dim=-1
            )
            for _ in range(500):
                env.step(action)
            # The five-second hold must preserve both the tool and its contents.
            assert ((spoon.data.root_link_pos_w.torch - initial_spoon[:, :3]).norm(dim=-1) < 0.01).all()
            pose = spoon.data.root_link_pose_w.torch
            retained = quat_apply_inverse(
                pose[:, None, 3:].expand(-1, count, -1), coffee.data.particle_pos_w.torch[:, :count] - pose[:, None, :3]
            )
            radius_squared = (retained[..., 0] / (SPOON_BOWL_LENGTH / 2.0)).square() + (
                retained[..., 1] / (SPOON_BOWL_WIDTH / 2.0)
            ).square()
            inside = (
                (radius_squared < 1.0)
                & (retained[..., 2] > SPOON_BOWL_DEPTH * radius_squared - 0.0005)
                & (retained[..., 2] < SPOON_BOWL_DEPTH + 0.006)
            )
            assert (inside.float().mean(dim=1) > 0.99).all(), "Coffee leaked from a stationary contact-held spoon."
            opened = action.clone()
            for name in ("right_thumb_MCP_FE", "right_thumb_IP", "right_index_PIP", "right_index_DIP"):
                opened[1, 7 + HAND_JOINT_NAMES.index(name)] = 0.0
            for step in range(40):
                targets = torch.lerp(action, opened, min((step + 1) / 20.0, 1.0))
                env.step(targets)
            actual = spoon.data.root_link_pose_w.torch.clone()
            palm = robot.data.body_link_pose_w.torch[:, palm_id]
            offset, _ = subtract_frame_transforms(palm[:, :3], palm[:, 3:], actual[:, :3], actual[:, 3:])
            assert torch.isfinite(actual).all()
            assert (offset[0] - initial_offset[0]).norm() < 0.01
            assert actual[1, 2] < initial_spoon[1, 2] - 0.05

            # Reset must restore the released spoon and its proxy without moving
            # the tool that is still held in the other world.
            env.reset(env_ids=torch.tensor([1], device=device))
            torch.testing.assert_close(spoon.data.root_link_pose_w.torch[0], actual[0], rtol=0, atol=0)
            torch.testing.assert_close(spoon.data.root_link_pose_w.torch[1], initial_spoon[1], rtol=0, atol=1.0e-6)
            for _ in range(20):
                env.step(action)
            assert ((spoon.data.root_link_pos_w.torch - initial_spoon[:, :3]).norm(dim=-1) < 0.01).all()
        finally:
            env.close()
