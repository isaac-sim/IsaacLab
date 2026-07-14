# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import warp as wp
from isaaclab_newton.physics import NewtonCfg
from isaaclab_physx.physics import PhysxCfg

from isaaclab_tasks.core.locomotion.ant.ant_direct_env import AntEnv
from isaaclab_tasks.core.locomotion.ant.ant_post_step import _AntPostStepBuffers
from isaaclab_tasks.core.locomotion.locomotion_direct_env import (
    compute_intermediate_values,
    compute_rewards,
    normalize_angle,
)

_NUM_ENVS = 8
_NUM_JOINTS = 8
_OBSERVATION_SIZE = 12 + 3 * _NUM_JOINTS


def _make_environment_data(device: str) -> SimpleNamespace:
    dtype = torch.float32
    angles = torch.tensor(
        [0.0, torch.pi - 1.0e-5, -torch.pi + 1.0e-5, 0.5, -0.7, 1.2, -2.1, 0.9],
        dtype=dtype,
        device=device,
    )
    half_angles = 0.5 * angles
    torso_rotation = torch.zeros((_NUM_ENVS, 4), dtype=dtype, device=device)
    torso_rotation[:, 2] = torch.sin(half_angles)
    torso_rotation[:, 3] = torch.cos(half_angles)
    # Exercise the pitch saturation branch with an exactly normalized 90-degree rotation.
    torso_rotation[-1] = torch.tensor([0.0, 2.0**-0.5, 0.0, 2.0**-0.5], dtype=dtype, device=device)

    torso_position = torch.tensor(
        [
            [0.0, 0.0, 0.31 - 1.0e-5],
            [2.0, -1.0, 0.31],
            [-2.0, 3.0, 0.31 + 1.0e-5],
            [0.5, 0.25, 0.8],
            [-0.5, -0.25, -0.2],
            [4.0, -3.0, 1.0],
            [-4.0, 3.0, 0.5],
            [1.0, 2.0, 0.4],
        ],
        dtype=dtype,
        device=device,
    )
    targets = torso_position.clone()
    targets[:, 0] += torch.tensor([1000.0, 0.0, -4.0, 2.0, -3.0, 1.0, -1.0, 5.0], device=device)
    targets[:, 1] += torch.tensor([0.0, 0.0, 3.0, -2.0, 4.0, -1.0, 1.0, 2.0], device=device)
    targets[:, 2] += 7.0  # Stable task semantics ignore target height.

    values = torch.arange(_NUM_ENVS * 3, dtype=dtype, device=device).reshape(_NUM_ENVS, 3)
    velocity = (values - 8.0) * 0.17
    angular_velocity = (11.0 - values) * 0.13
    joint_position = torch.linspace(-1.2, 1.2, _NUM_ENVS * _NUM_JOINTS, device=device).reshape(_NUM_ENVS, _NUM_JOINTS)
    joint_velocity = torch.linspace(2.0, -2.0, _NUM_ENVS * _NUM_JOINTS, device=device).reshape(_NUM_ENVS, _NUM_JOINTS)
    actions = torch.linspace(-1.5, 1.5, _NUM_ENVS * _NUM_JOINTS, device=device).reshape(_NUM_ENVS, _NUM_JOINTS)

    cfg = SimpleNamespace(
        sim=SimpleNamespace(dt=1.0 / 120.0),
        angular_velocity_scale=1.0,
        dof_vel_scale=0.2,
        termination_height=0.31,
        up_weight=0.1,
        heading_weight=0.5,
        actions_cost_scale=0.005,
        energy_cost_scale=0.05,
        death_cost=-2.0,
        alive_reward_scale=0.5,
    )
    return SimpleNamespace(
        cfg=cfg,
        targets=targets,
        torso_position=torso_position,
        torso_rotation=torso_rotation,
        velocity=velocity,
        ang_velocity=angular_velocity,
        dof_pos=joint_position,
        dof_vel=joint_velocity,
        _joint_position_lower_limits=torch.linspace(-1.4, -0.7, _NUM_JOINTS, device=device),
        _joint_position_upper_limits=torch.linspace(0.8, 1.5, _NUM_JOINTS, device=device),
        _joint_position_limits=wp.from_torch(
            torch.stack(
                (
                    torch.linspace(-1.4, -0.7, _NUM_JOINTS, device=device).expand(_NUM_ENVS, -1),
                    torch.linspace(0.8, 1.5, _NUM_JOINTS, device=device).expand(_NUM_ENVS, -1),
                ),
                dim=-1,
            ).contiguous(),
            dtype=wp.vec2f,
        ),
        actions=actions,
        episode_length_buf=torch.tensor([0, 898, 899, 900, 10, 899, 1, 50], dtype=torch.int64, device=device),
        motor_effort_ratio=torch.linspace(0.7, 1.3, _NUM_JOINTS, device=device),
        max_episode_length=900,
        potentials=torch.linspace(-100.0, -107.0, _NUM_ENVS, device=device),
        prev_potentials=torch.linspace(-90.0, -97.0, _NUM_ENVS, device=device),
    )


def _compute_torch_reference(env: SimpleNamespace) -> dict[str, torch.Tensor]:
    inv_start_rotation = torch.tensor([0.0, 0.0, 0.0, 1.0], device=env.torso_position.device).repeat(_NUM_ENVS, 1)
    heading_basis = torch.tensor([1.0, 0.0, 0.0], device=env.torso_position.device).repeat(_NUM_ENVS, 1)
    up_basis = torch.tensor([0.0, 0.0, 1.0], device=env.torso_position.device).repeat(_NUM_ENVS, 1)
    potentials = env.potentials.clone()
    previous_potentials = env.prev_potentials.clone()
    (
        up_projection,
        heading_projection,
        up_vector,
        heading_vector,
        local_velocity,
        local_angular_velocity,
        roll,
        pitch,
        yaw,
        angle_to_target,
        scaled_joint_position,
        previous_potentials,
        potentials,
    ) = compute_intermediate_values(
        env.targets,
        env.torso_position,
        env.torso_rotation,
        env.velocity,
        env.ang_velocity,
        env.dof_pos,
        env._joint_position_lower_limits,
        env._joint_position_upper_limits,
        inv_start_rotation,
        heading_basis,
        up_basis,
        potentials,
        previous_potentials,
        env.cfg.sim.dt,
    )
    terminated = env.torso_position[:, 2] < env.cfg.termination_height
    time_out = env.episode_length_buf >= env.max_episode_length - 1
    reward = compute_rewards(
        env.actions,
        terminated,
        env.cfg.up_weight,
        env.cfg.heading_weight,
        heading_projection,
        up_projection,
        env.dof_vel,
        scaled_joint_position,
        potentials,
        previous_potentials,
        env.cfg.actions_cost_scale,
        env.cfg.energy_cost_scale,
        env.cfg.dof_vel_scale,
        env.cfg.death_cost,
        env.cfg.alive_reward_scale,
        env.motor_effort_ratio,
    )
    observation = torch.cat(
        (
            env.torso_position[:, 2:3],
            local_velocity,
            local_angular_velocity * env.cfg.angular_velocity_scale,
            normalize_angle(yaw).unsqueeze(-1),
            normalize_angle(roll).unsqueeze(-1),
            normalize_angle(angle_to_target).unsqueeze(-1),
            up_projection.unsqueeze(-1),
            heading_projection.unsqueeze(-1),
            scaled_joint_position,
            env.dof_vel * env.cfg.dof_vel_scale,
            env.actions,
        ),
        dim=-1,
    )
    return {
        "up_projection": up_projection,
        "heading_projection": heading_projection,
        "up_vector": up_vector,
        "heading_vector": heading_vector,
        "local_velocity": local_velocity,
        "local_angular_velocity": local_angular_velocity,
        "roll": roll,
        "pitch": pitch,
        "yaw": yaw,
        "angle_to_target": angle_to_target,
        "scaled_joint_position": scaled_joint_position,
        "previous_potentials": previous_potentials,
        "potentials": potentials,
        "terminated": terminated,
        "time_out": time_out,
        "reward": reward,
        "observation": observation,
    }


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_fused_post_step_matches_torch_reference(device: str):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    env = _make_environment_data(device)
    reference = _compute_torch_reference(env)
    buffers = _AntPostStepBuffers(_NUM_ENVS, _NUM_JOINTS, _OBSERVATION_SIZE, device)
    buffers.bind_environment_outputs(env)
    buffers.compute_post_step(env)
    wp.synchronize_device(device)

    actual = {
        "up_projection": buffers.up_projection_torch,
        "heading_projection": buffers.heading_projection_torch,
        "up_vector": buffers.up_vector_torch,
        "heading_vector": buffers.heading_vector_torch,
        "local_velocity": buffers.local_velocity_torch,
        "local_angular_velocity": buffers.local_angular_velocity_torch,
        "roll": buffers.roll_torch,
        "pitch": buffers.pitch_torch,
        "yaw": buffers.yaw_torch,
        "angle_to_target": buffers.angle_to_target_torch,
        "scaled_joint_position": buffers.scaled_joint_position_torch,
        "previous_potentials": env.prev_potentials,
        "potentials": env.potentials,
        "terminated": buffers.terminated_torch,
        "time_out": buffers.time_out_torch,
        "reward": buffers.reward_torch,
        "observation": buffers.observation_torch,
    }
    for name, expected in reference.items():
        torch.testing.assert_close(actual[name], expected, rtol=2.0e-5, atol=2.0e-5, equal_nan=True)


def test_post_reset_refresh_preserves_terminal_outputs():
    env = _make_environment_data("cpu")
    buffers = _AntPostStepBuffers(_NUM_ENVS, _NUM_JOINTS, _OBSERVATION_SIZE, "cpu")
    buffers.bind_environment_outputs(env)
    buffers.compute_post_step(env)
    wp.synchronize_device("cpu")
    reward = buffers.reward_torch.clone()
    terminated = buffers.terminated_torch.clone()
    time_out = buffers.time_out_torch.clone()

    env.torso_position[:, 0] += 0.25
    env.velocity.mul_(0.5)
    env.actions.neg_()
    reference = _compute_torch_reference(env)
    buffers.compute_intermediate_and_observation(env)
    wp.synchronize_device("cpu")

    torch.testing.assert_close(buffers.reward_torch, reward)
    torch.testing.assert_close(buffers.terminated_torch, terminated)
    torch.testing.assert_close(buffers.time_out_torch, time_out)
    torch.testing.assert_close(buffers.observation_torch, reference["observation"], rtol=2.0e-5, atol=2.0e-5)
    torch.testing.assert_close(env.potentials, reference["potentials"], rtol=2.0e-5, atol=2.0e-5)
    torch.testing.assert_close(env.prev_potentials, reference["previous_potentials"], rtol=2.0e-5, atol=2.0e-5)


def test_masked_reset_refresh_updates_only_selected_environments():
    env = _make_environment_data("cpu")
    buffers = _AntPostStepBuffers(_NUM_ENVS, _NUM_JOINTS, _OBSERVATION_SIZE, "cpu")
    buffers.bind_environment_outputs(env)
    buffers.compute_post_step(env)
    wp.synchronize_device("cpu")

    env_mask = torch.tensor([True, False, False, True, False, True, False, False])
    env.torso_position[env_mask] = torch.tensor(
        [[0.25, -0.5, 0.5], [1.5, 0.75, 0.5], [-2.0, 1.0, 0.5]],
        dtype=torch.float32,
    )
    env.velocity[env_mask] = 0.0
    env.ang_velocity[env_mask] = 0.0
    env.dof_pos[env_mask] = 0.0
    env.dof_vel[env_mask] = 0.0

    observation_before = buffers.observation_torch.clone()
    potentials_before = env.potentials.clone()
    previous_potentials_before = env.prev_potentials.clone()
    reward_before = buffers.reward_torch.clone()
    terminated_before = buffers.terminated_torch.clone()
    time_out_before = buffers.time_out_torch.clone()
    episode_length_before = env.episode_length_buf.clone()

    reference_env = SimpleNamespace(**vars(env))
    reference_env.potentials = env.potentials.clone()
    target_delta = env.targets[env_mask, :2] - env.torso_position[env_mask, :2]
    reference_env.potentials[env_mask] = -torch.linalg.norm(target_delta, dim=-1) / env.cfg.sim.dt
    reference = _compute_torch_reference(reference_env)

    buffers.compute_masked_reset_observation(env, wp.from_torch(env_mask))
    wp.synchronize_device("cpu")

    torch.testing.assert_close(
        buffers.observation_torch[env_mask], reference["observation"][env_mask], rtol=2e-5, atol=2e-5
    )
    torch.testing.assert_close(env.potentials[env_mask], reference["potentials"][env_mask], rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(
        env.prev_potentials[env_mask], reference["previous_potentials"][env_mask], rtol=2e-5, atol=2e-5
    )
    torch.testing.assert_close(buffers.observation_torch[~env_mask], observation_before[~env_mask])
    torch.testing.assert_close(env.potentials[~env_mask], potentials_before[~env_mask])
    torch.testing.assert_close(env.prev_potentials[~env_mask], previous_potentials_before[~env_mask])
    torch.testing.assert_close(buffers.reward_torch, reward_before)
    torch.testing.assert_close(buffers.terminated_torch, terminated_before)
    torch.testing.assert_close(buffers.time_out_torch, time_out_before)
    torch.testing.assert_close(env.episode_length_buf[env_mask], torch.zeros(3, dtype=torch.int64))
    torch.testing.assert_close(env.episode_length_buf[~env_mask], episode_length_before[~env_mask])


@pytest.mark.parametrize(
    ("physics", "compute_final_obs"),
    [(PhysxCfg(), False), (NewtonCfg(), True)],
)
def test_fused_post_step_falls_back_for_unsupported_configurations(
    physics, compute_final_obs: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = object.__new__(AntEnv)
    env._is_closed = True
    env.cfg = SimpleNamespace(sim=SimpleNamespace(physics=physics), compute_final_obs=compute_final_obs)
    env._use_fused_post_step = env._supports_fused_post_step()
    expected = (torch.tensor([True]), torch.tensor([False]))
    fallback = Mock(return_value=expected)
    monkeypatch.setattr("isaaclab_tasks.core.locomotion.locomotion_direct_env.LocomotionDirectEnv._get_dones", fallback)

    assert not env._use_fused_post_step
    assert env._get_dones() is expected
    fallback.assert_called_once_with()


def test_fused_post_step_selects_newton_without_final_observations() -> None:
    env = object.__new__(AntEnv)
    env._is_closed = True
    env.cfg = SimpleNamespace(sim=SimpleNamespace(physics=NewtonCfg()), compute_final_obs=False)

    assert env._supports_fused_post_step()


def test_masked_reset_preserves_robot_buffers_and_metrics() -> None:
    env = object.__new__(AntEnv)
    env._is_closed = True
    env.sim = SimpleNamespace(device="cpu")
    env.reset_buf = torch.tensor([False, True, False, True])
    env.reset_time_outs = torch.tensor([False, True, False, False])
    env.extras = {}
    instantaneous_wrench_composer = Mock(active=True)
    permanent_wrench_composer = Mock(active=True)
    env.robot = Mock(
        instantaneous_wrench_composer=instantaneous_wrench_composer,
        permanent_wrench_composer=permanent_wrench_composer,
    )
    env.robot.data = SimpleNamespace(
        default_root_vel=SimpleNamespace(warp=object()),
        default_joint_pos=SimpleNamespace(warp=object()),
        default_joint_vel=SimpleNamespace(warp=object()),
    )
    env._default_root_pose_w = object()
    env._post_step_buffers = Mock()
    env_mask = wp.from_torch(env.reset_buf)

    env._reset_idx_mask(env_mask)

    torch.testing.assert_close(env.extras["log"]["Metrics/success_rate"], torch.tensor(0.5))
    instantaneous_wrench_composer.reset.assert_called_once_with(env_mask=env_mask)
    permanent_wrench_composer.reset.assert_called_once_with(env_mask=env_mask)
    env.robot.write_root_pose_to_sim_mask.assert_called_once_with(root_pose=env._default_root_pose_w, env_mask=env_mask)
    env.robot.write_root_velocity_to_sim_mask.assert_called_once_with(
        root_velocity=env.robot.data.default_root_vel.warp, env_mask=env_mask
    )
    env.robot.write_joint_state_to_sim_mask.assert_called_once_with(
        position=env.robot.data.default_joint_pos.warp,
        velocity=env.robot.data.default_joint_vel.warp,
        env_mask=env_mask,
    )
    env._post_step_buffers.compute_masked_reset_observation.assert_called_once_with(env, env_mask)
