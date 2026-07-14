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

from isaaclab_tasks.core.locomotion.locomotion_direct_env import (
    LocomotionDirectEnv,
    compute_intermediate_values,
    compute_rewards,
    normalize_angle,
)
from isaaclab_tasks.core.locomotion.locomotion_post_step import _LocomotionPostStepBuffers

_NUM_ENVS = 8
_NUM_JOINTS = 8
_OBSERVATION_SIZE = 12 + 3 * _NUM_JOINTS


def _make_environment_data(device: str, num_joints: int = _NUM_JOINTS) -> SimpleNamespace:
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
    joint_position = torch.linspace(-1.2, 1.2, _NUM_ENVS * num_joints, device=device).reshape(_NUM_ENVS, num_joints)
    joint_velocity = torch.linspace(2.0, -2.0, _NUM_ENVS * num_joints, device=device).reshape(_NUM_ENVS, num_joints)
    actions = torch.linspace(-1.5, 1.5, _NUM_ENVS * num_joints, device=device).reshape(_NUM_ENVS, num_joints)

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
        _joint_position_lower_limits=torch.linspace(-1.4, -0.7, num_joints, device=device),
        _joint_position_upper_limits=torch.linspace(0.8, 1.5, num_joints, device=device),
        _joint_position_limits=wp.from_torch(
            torch.stack(
                (
                    torch.linspace(-1.4, -0.7, num_joints, device=device).expand(_NUM_ENVS, -1),
                    torch.linspace(0.8, 1.5, num_joints, device=device).expand(_NUM_ENVS, -1),
                ),
                dim=-1,
            ).contiguous(),
            dtype=wp.vec2f,
        ),
        actions=actions,
        episode_length_buf=torch.tensor([0, 898, 899, 900, 10, 899, 1, 50], dtype=torch.int64, device=device),
        motor_effort_ratio=torch.linspace(0.7, 1.3, num_joints, device=device),
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
@pytest.mark.parametrize("num_joints", [8, 21])
def test_fused_post_step_matches_torch_reference(device: str, num_joints: int):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    env = _make_environment_data(device, num_joints)
    reference = _compute_torch_reference(env)
    buffers = _LocomotionPostStepBuffers(_NUM_ENVS, num_joints, 12 + 3 * num_joints, device)
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


def test_post_step_preserves_previous_observation():
    env = _make_environment_data("cpu")
    buffers = _LocomotionPostStepBuffers(_NUM_ENVS, _NUM_JOINTS, _OBSERVATION_SIZE, "cpu")
    buffers.compute_post_step(env)
    wp.synchronize_device("cpu")
    previous_observation = buffers.observation_torch
    expected_previous_observation = previous_observation.clone()

    env.torso_position[:, 0] += 0.25
    buffers.compute_post_step(env)
    wp.synchronize_device("cpu")

    assert buffers.observation_torch.data_ptr() != previous_observation.data_ptr()
    torch.testing.assert_close(previous_observation, expected_previous_observation)


def test_post_reset_refresh_preserves_terminal_outputs():
    env = _make_environment_data("cpu")
    buffers = _LocomotionPostStepBuffers(_NUM_ENVS, _NUM_JOINTS, _OBSERVATION_SIZE, "cpu")
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


@pytest.mark.parametrize(
    ("physics", "compute_final_obs"),
    [(PhysxCfg(), True), (NewtonCfg(), True)],
)
def test_fused_post_step_falls_back_for_unsupported_configurations(physics, compute_final_obs: bool) -> None:
    env = object.__new__(LocomotionDirectEnv)
    env._is_closed = True
    env.cfg = SimpleNamespace(
        sim=SimpleNamespace(physics=physics, dt=0.1),
        action_space=8,
        observation_space=36,
        compute_final_obs=compute_final_obs,
        decimation=1,
        episode_length_s=1.0,
        termination_height=0.31,
    )
    env.robot = SimpleNamespace(num_joints=8)
    env._use_fused_post_step = env._supports_fused_post_step()
    env._compute_intermediate_values = Mock()
    env.episode_length_buf = torch.tensor([0, 9])
    env.torso_position = torch.tensor([[0.0, 0.0, 0.2], [0.0, 0.0, 1.0]])

    assert not env._use_fused_post_step
    terminated, time_out = env._get_dones()
    env._compute_intermediate_values.assert_called_once_with()
    torch.testing.assert_close(terminated, torch.tensor([True, False]))
    torch.testing.assert_close(time_out, torch.tensor([False, True]))


@pytest.mark.parametrize("physics", [NewtonCfg(), PhysxCfg()])
def test_fused_post_step_selects_supported_backend_without_final_observations(physics) -> None:
    env = object.__new__(LocomotionDirectEnv)
    env._is_closed = True
    env.cfg = SimpleNamespace(
        sim=SimpleNamespace(physics=physics), action_space=21, observation_space=75, compute_final_obs=False
    )
    env.robot = SimpleNamespace(num_joints=21)

    assert env._supports_fused_post_step()


@pytest.mark.parametrize(("action_space", "observation_space"), [(7, 36), (8, 35)])
def test_fused_post_step_rejects_incompatible_space_layout(action_space: int, observation_space: int) -> None:
    env = object.__new__(LocomotionDirectEnv)
    env._is_closed = True
    env.cfg = SimpleNamespace(
        sim=SimpleNamespace(physics=NewtonCfg()),
        action_space=action_space,
        observation_space=observation_space,
        compute_final_obs=False,
    )
    env.robot = SimpleNamespace(num_joints=8)

    assert not env._supports_fused_post_step()


def test_physx_fused_post_step_refreshes_pull_on_demand_inputs() -> None:
    accesses = []

    class PullOnDemandData:
        @property
        def root_link_pose_w(self):
            accesses.append("root_link_pose_w")

        @property
        def root_com_vel_w(self):
            accesses.append("root_com_vel_w")

        @property
        def joint_pos(self):
            accesses.append("joint_pos")

        @property
        def joint_vel(self):
            accesses.append("joint_vel")

    post_step_buffers = SimpleNamespace(
        compute_post_step=lambda env: accesses.append("compute_post_step"),
        terminated_torch=torch.tensor([False]),
        time_out_torch=torch.tensor([False]),
    )
    env = object.__new__(LocomotionDirectEnv)
    env._is_closed = True
    env._use_fused_post_step = True
    env._fused_inputs_require_refresh = True
    env.robot = SimpleNamespace(data=PullOnDemandData())
    env._post_step_buffers = post_step_buffers

    env._get_dones()

    assert accesses == ["root_link_pose_w", "root_com_vel_w", "joint_pos", "joint_vel", "compute_post_step"]
