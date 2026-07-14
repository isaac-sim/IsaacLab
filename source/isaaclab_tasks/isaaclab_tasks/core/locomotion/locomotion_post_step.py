# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import warp as wp


@wp.func
def _quat_apply_xyzw(quat: wp.vec4f, vector: wp.vec3f) -> wp.vec3f:
    quat_xyz = wp.vec3f(quat[0], quat[1], quat[2])
    cross = 2.0 * wp.cross(quat_xyz, vector)
    return vector + quat[3] * cross + wp.cross(quat_xyz, cross)


@wp.func
def _quat_apply_inverse_xyzw(quat: wp.vec4f, vector: wp.vec3f) -> wp.vec3f:
    quat_xyz = wp.vec3f(quat[0], quat[1], quat[2])
    cross = 2.0 * wp.cross(quat_xyz, vector)
    return vector - quat[3] * cross + wp.cross(quat_xyz, cross)


@wp.func
def _normalize_angle(angle: wp.float32) -> wp.float32:
    return wp.atan2(wp.sin(angle), wp.cos(angle))


@wp.func
def _compute_intermediate_and_observation(
    env_id: int,
    targets: wp.array2d(dtype=wp.float32),
    torso_position: wp.array2d(dtype=wp.float32),
    torso_rotation: wp.array2d(dtype=wp.float32),
    velocity: wp.array2d(dtype=wp.float32),
    angular_velocity: wp.array2d(dtype=wp.float32),
    joint_position: wp.array2d(dtype=wp.float32),
    joint_velocity: wp.array2d(dtype=wp.float32),
    joint_position_limits: wp.array2d(dtype=wp.vec2f),
    actions: wp.array2d(dtype=wp.float32),
    dt: wp.float32,
    angular_velocity_scale: wp.float32,
    joint_velocity_scale: wp.float32,
    num_joints: int,
    potentials: wp.array(dtype=wp.float32),
    previous_potentials: wp.array(dtype=wp.float32),
    up_projection: wp.array(dtype=wp.float32),
    heading_projection: wp.array(dtype=wp.float32),
    up_vector: wp.array2d(dtype=wp.float32),
    heading_vector: wp.array2d(dtype=wp.float32),
    local_velocity: wp.array2d(dtype=wp.float32),
    local_angular_velocity: wp.array2d(dtype=wp.float32),
    roll: wp.array(dtype=wp.float32),
    pitch: wp.array(dtype=wp.float32),
    yaw: wp.array(dtype=wp.float32),
    angle_to_target: wp.array(dtype=wp.float32),
    scaled_joint_position: wp.array2d(dtype=wp.float32),
    observation: wp.array2d(dtype=wp.float32),
):
    position = wp.vec3f(torso_position[env_id, 0], torso_position[env_id, 1], torso_position[env_id, 2])
    quat = wp.vec4f(
        torso_rotation[env_id, 0],
        torso_rotation[env_id, 1],
        torso_rotation[env_id, 2],
        torso_rotation[env_id, 3],
    )
    target_delta = wp.vec3f(
        targets[env_id, 0] - position[0],
        targets[env_id, 1] - position[1],
        0.0,
    )
    target_distance = wp.length(target_delta)
    target_direction = target_delta / wp.max(target_distance, 1.0e-9)

    up = _quat_apply_xyzw(quat, wp.vec3f(0.0, 0.0, 1.0))
    heading = _quat_apply_xyzw(quat, wp.vec3f(1.0, 0.0, 0.0))
    up_proj = up[2]
    heading_proj = wp.dot(heading, target_direction)

    world_velocity = wp.vec3f(velocity[env_id, 0], velocity[env_id, 1], velocity[env_id, 2])
    world_angular_velocity = wp.vec3f(
        angular_velocity[env_id, 0], angular_velocity[env_id, 1], angular_velocity[env_id, 2]
    )
    velocity_local = _quat_apply_inverse_xyzw(quat, world_velocity)
    angular_velocity_local = _quat_apply_inverse_xyzw(quat, world_angular_velocity)

    quat_x = quat[0]
    quat_y = quat[1]
    quat_z = quat[2]
    quat_w = quat[3]
    sin_roll = 2.0 * (quat_w * quat_x + quat_y * quat_z)
    cos_roll = 1.0 - 2.0 * (quat_x * quat_x + quat_y * quat_y)
    roll_value = wp.atan2(sin_roll, cos_roll)
    sin_pitch = 2.0 * (quat_w * quat_y - quat_z * quat_x)
    pitch_value = wp.asin(sin_pitch)
    if wp.abs(sin_pitch) >= 1.0:
        pitch_value = wp.sign(sin_pitch) * wp.pi * 0.5
    sin_yaw = 2.0 * (quat_w * quat_z + quat_x * quat_y)
    cos_yaw = 1.0 - 2.0 * (quat_y * quat_y + quat_z * quat_z)
    yaw_value = wp.atan2(sin_yaw, cos_yaw)
    target_angle = wp.atan2(target_delta[1], target_delta[0]) - yaw_value

    previous_potentials[env_id] = potentials[env_id]
    potentials[env_id] = -target_distance / dt
    up_projection[env_id] = up_proj
    heading_projection[env_id] = heading_proj
    up_vector[env_id, 0] = up[0]
    up_vector[env_id, 1] = up[1]
    up_vector[env_id, 2] = up[2]
    heading_vector[env_id, 0] = heading[0]
    heading_vector[env_id, 1] = heading[1]
    heading_vector[env_id, 2] = heading[2]
    local_velocity[env_id, 0] = velocity_local[0]
    local_velocity[env_id, 1] = velocity_local[1]
    local_velocity[env_id, 2] = velocity_local[2]
    local_angular_velocity[env_id, 0] = angular_velocity_local[0]
    local_angular_velocity[env_id, 1] = angular_velocity_local[1]
    local_angular_velocity[env_id, 2] = angular_velocity_local[2]
    roll[env_id] = roll_value
    pitch[env_id] = pitch_value
    yaw[env_id] = yaw_value
    angle_to_target[env_id] = target_angle

    observation[env_id, 0] = position[2]
    observation[env_id, 1] = velocity_local[0]
    observation[env_id, 2] = velocity_local[1]
    observation[env_id, 3] = velocity_local[2]
    observation[env_id, 4] = angular_velocity_local[0] * angular_velocity_scale
    observation[env_id, 5] = angular_velocity_local[1] * angular_velocity_scale
    observation[env_id, 6] = angular_velocity_local[2] * angular_velocity_scale
    observation[env_id, 7] = _normalize_angle(yaw_value)
    observation[env_id, 8] = _normalize_angle(roll_value)
    observation[env_id, 9] = _normalize_angle(target_angle)
    observation[env_id, 10] = up_proj
    observation[env_id, 11] = heading_proj

    joint_velocity_offset = 12 + num_joints
    action_offset = joint_velocity_offset + num_joints
    for joint_id in range(num_joints):
        lower_limit = joint_position_limits[env_id, joint_id][0]
        upper_limit = joint_position_limits[env_id, joint_id][1]
        limit_offset = 0.5 * (lower_limit + upper_limit)
        scaled_position = 2.0 * (joint_position[env_id, joint_id] - limit_offset) / (upper_limit - lower_limit)
        scaled_joint_position[env_id, joint_id] = scaled_position
        observation[env_id, 12 + joint_id] = scaled_position
        observation[env_id, joint_velocity_offset + joint_id] = joint_velocity[env_id, joint_id] * joint_velocity_scale
        observation[env_id, action_offset + joint_id] = actions[env_id, joint_id]


@wp.kernel
def compute_locomotion_post_step(
    targets: wp.array2d(dtype=wp.float32),
    torso_position: wp.array2d(dtype=wp.float32),
    torso_rotation: wp.array2d(dtype=wp.float32),
    velocity: wp.array2d(dtype=wp.float32),
    angular_velocity: wp.array2d(dtype=wp.float32),
    joint_position: wp.array2d(dtype=wp.float32),
    joint_velocity: wp.array2d(dtype=wp.float32),
    joint_position_limits: wp.array2d(dtype=wp.vec2f),
    actions: wp.array2d(dtype=wp.float32),
    episode_length: wp.array(dtype=wp.int64),
    motor_effort_ratio: wp.array(dtype=wp.float32),
    dt: wp.float32,
    angular_velocity_scale: wp.float32,
    joint_velocity_scale: wp.float32,
    termination_height: wp.float32,
    max_episode_length: wp.int64,
    up_weight: wp.float32,
    heading_weight: wp.float32,
    actions_cost_scale: wp.float32,
    energy_cost_scale: wp.float32,
    death_cost: wp.float32,
    alive_reward_scale: wp.float32,
    num_joints: int,
    potentials: wp.array(dtype=wp.float32),
    previous_potentials: wp.array(dtype=wp.float32),
    up_projection: wp.array(dtype=wp.float32),
    heading_projection: wp.array(dtype=wp.float32),
    up_vector: wp.array2d(dtype=wp.float32),
    heading_vector: wp.array2d(dtype=wp.float32),
    local_velocity: wp.array2d(dtype=wp.float32),
    local_angular_velocity: wp.array2d(dtype=wp.float32),
    roll: wp.array(dtype=wp.float32),
    pitch: wp.array(dtype=wp.float32),
    yaw: wp.array(dtype=wp.float32),
    angle_to_target: wp.array(dtype=wp.float32),
    scaled_joint_position: wp.array2d(dtype=wp.float32),
    terminated: wp.array(dtype=wp.bool),
    time_out: wp.array(dtype=wp.bool),
    reward: wp.array(dtype=wp.float32),
    observation: wp.array2d(dtype=wp.float32),
):
    env_id = wp.tid()
    _compute_intermediate_and_observation(
        env_id,
        targets,
        torso_position,
        torso_rotation,
        velocity,
        angular_velocity,
        joint_position,
        joint_velocity,
        joint_position_limits,
        actions,
        dt,
        angular_velocity_scale,
        joint_velocity_scale,
        num_joints,
        potentials,
        previous_potentials,
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
        observation,
    )

    died = torso_position[env_id, 2] < termination_height
    terminated[env_id] = died
    time_out[env_id] = episode_length[env_id] >= max_episode_length - wp.int64(1)

    heading_reward = heading_weight * heading_projection[env_id] / 0.8
    if heading_projection[env_id] > 0.8:
        heading_reward = heading_weight
    up_reward = wp.float32(0.0)
    if up_projection[env_id] > 0.93:
        up_reward = up_weight

    actions_cost = wp.float32(0.0)
    electricity_cost = wp.float32(0.0)
    joint_limit_cost = wp.float32(0.0)
    for joint_id in range(num_joints):
        action = actions[env_id, joint_id]
        actions_cost += action * action
        electricity_cost += (
            wp.abs(action * joint_velocity[env_id, joint_id] * joint_velocity_scale) * motor_effort_ratio[joint_id]
        )
        if scaled_joint_position[env_id, joint_id] > 0.98:
            joint_limit_cost += 1.0

    total_reward = (
        potentials[env_id]
        - previous_potentials[env_id]
        + alive_reward_scale
        + up_reward
        + heading_reward
        - actions_cost_scale * actions_cost
        - energy_cost_scale * electricity_cost
        - joint_limit_cost
    )
    if died:
        total_reward = death_cost
    reward[env_id] = total_reward


@wp.kernel
def compute_locomotion_intermediate_and_observation(
    targets: wp.array2d(dtype=wp.float32),
    torso_position: wp.array2d(dtype=wp.float32),
    torso_rotation: wp.array2d(dtype=wp.float32),
    velocity: wp.array2d(dtype=wp.float32),
    angular_velocity: wp.array2d(dtype=wp.float32),
    joint_position: wp.array2d(dtype=wp.float32),
    joint_velocity: wp.array2d(dtype=wp.float32),
    joint_position_limits: wp.array2d(dtype=wp.vec2f),
    actions: wp.array2d(dtype=wp.float32),
    dt: wp.float32,
    angular_velocity_scale: wp.float32,
    joint_velocity_scale: wp.float32,
    num_joints: int,
    potentials: wp.array(dtype=wp.float32),
    previous_potentials: wp.array(dtype=wp.float32),
    up_projection: wp.array(dtype=wp.float32),
    heading_projection: wp.array(dtype=wp.float32),
    up_vector: wp.array2d(dtype=wp.float32),
    heading_vector: wp.array2d(dtype=wp.float32),
    local_velocity: wp.array2d(dtype=wp.float32),
    local_angular_velocity: wp.array2d(dtype=wp.float32),
    roll: wp.array(dtype=wp.float32),
    pitch: wp.array(dtype=wp.float32),
    yaw: wp.array(dtype=wp.float32),
    angle_to_target: wp.array(dtype=wp.float32),
    scaled_joint_position: wp.array2d(dtype=wp.float32),
    observation: wp.array2d(dtype=wp.float32),
):
    env_id = wp.tid()
    _compute_intermediate_and_observation(
        env_id,
        targets,
        torso_position,
        torso_rotation,
        velocity,
        angular_velocity,
        joint_position,
        joint_velocity,
        joint_position_limits,
        actions,
        dt,
        angular_velocity_scale,
        joint_velocity_scale,
        num_joints,
        potentials,
        previous_potentials,
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
        observation,
    )


@wp.kernel
def compute_locomotion_masked_reset_observation(
    env_mask: wp.array(dtype=wp.bool),
    episode_length: wp.array(dtype=wp.int64),
    targets: wp.array2d(dtype=wp.float32),
    torso_position: wp.array2d(dtype=wp.float32),
    torso_rotation: wp.array2d(dtype=wp.float32),
    velocity: wp.array2d(dtype=wp.float32),
    angular_velocity: wp.array2d(dtype=wp.float32),
    joint_position: wp.array2d(dtype=wp.float32),
    joint_velocity: wp.array2d(dtype=wp.float32),
    joint_position_limits: wp.array2d(dtype=wp.vec2f),
    actions: wp.array2d(dtype=wp.float32),
    dt: wp.float32,
    angular_velocity_scale: wp.float32,
    joint_velocity_scale: wp.float32,
    num_joints: int,
    potentials: wp.array(dtype=wp.float32),
    previous_potentials: wp.array(dtype=wp.float32),
    up_projection: wp.array(dtype=wp.float32),
    heading_projection: wp.array(dtype=wp.float32),
    up_vector: wp.array2d(dtype=wp.float32),
    heading_vector: wp.array2d(dtype=wp.float32),
    local_velocity: wp.array2d(dtype=wp.float32),
    local_angular_velocity: wp.array2d(dtype=wp.float32),
    roll: wp.array(dtype=wp.float32),
    pitch: wp.array(dtype=wp.float32),
    yaw: wp.array(dtype=wp.float32),
    angle_to_target: wp.array(dtype=wp.float32),
    scaled_joint_position: wp.array2d(dtype=wp.float32),
    observation: wp.array2d(dtype=wp.float32),
):
    env_id = wp.tid()
    if env_mask[env_id]:
        target_x = targets[env_id, 0] - torso_position[env_id, 0]
        target_y = targets[env_id, 1] - torso_position[env_id, 1]
        potentials[env_id] = -wp.sqrt(target_x * target_x + target_y * target_y) / dt
        episode_length[env_id] = wp.int64(0)
        _compute_intermediate_and_observation(
            env_id,
            targets,
            torso_position,
            torso_rotation,
            velocity,
            angular_velocity,
            joint_position,
            joint_velocity,
            joint_position_limits,
            actions,
            dt,
            angular_velocity_scale,
            joint_velocity_scale,
            num_joints,
            potentials,
            previous_potentials,
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
            observation,
        )


class _LocomotionPostStepBuffers:
    """Persistent task outputs used by the fused locomotion post-step kernels."""

    def __init__(self, num_envs: int, num_joints: int, observation_size: int, device: str):
        self.num_envs = num_envs
        self.num_joints = num_joints
        self.device = device

        self.up_projection = wp.zeros(num_envs, dtype=wp.float32, device=device)
        self.heading_projection = wp.zeros(num_envs, dtype=wp.float32, device=device)
        self.up_vector = wp.zeros((num_envs, 3), dtype=wp.float32, device=device)
        self.heading_vector = wp.zeros((num_envs, 3), dtype=wp.float32, device=device)
        self.local_velocity = wp.zeros((num_envs, 3), dtype=wp.float32, device=device)
        self.local_angular_velocity = wp.zeros((num_envs, 3), dtype=wp.float32, device=device)
        self.roll = wp.zeros(num_envs, dtype=wp.float32, device=device)
        self.pitch = wp.zeros(num_envs, dtype=wp.float32, device=device)
        self.yaw = wp.zeros(num_envs, dtype=wp.float32, device=device)
        self.angle_to_target = wp.zeros(num_envs, dtype=wp.float32, device=device)
        self.scaled_joint_position = wp.zeros((num_envs, num_joints), dtype=wp.float32, device=device)
        self.terminated = wp.zeros(num_envs, dtype=wp.bool, device=device)
        self.time_out = wp.zeros(num_envs, dtype=wp.bool, device=device)
        self.reward = wp.zeros(num_envs, dtype=wp.float32, device=device)
        # Keep two observation buffers because RL runners may retain the current
        # observation until after the next environment step completes.
        self._observations = (
            wp.zeros((num_envs, observation_size), dtype=wp.float32, device=device),
            wp.zeros((num_envs, observation_size), dtype=wp.float32, device=device),
        )
        self._observation_torch = tuple(wp.to_torch(observation) for observation in self._observations)
        self._observation_index = 0
        self.observation = self._observations[self._observation_index]
        self.observation_torch = self._observation_torch[self._observation_index]

        self.up_projection_torch = wp.to_torch(self.up_projection)
        self.heading_projection_torch = wp.to_torch(self.heading_projection)
        self.up_vector_torch = wp.to_torch(self.up_vector)
        self.heading_vector_torch = wp.to_torch(self.heading_vector)
        self.local_velocity_torch = wp.to_torch(self.local_velocity)
        self.local_angular_velocity_torch = wp.to_torch(self.local_angular_velocity)
        self.roll_torch = wp.to_torch(self.roll)
        self.pitch_torch = wp.to_torch(self.pitch)
        self.yaw_torch = wp.to_torch(self.yaw)
        self.angle_to_target_torch = wp.to_torch(self.angle_to_target)
        self.scaled_joint_position_torch = wp.to_torch(self.scaled_joint_position)
        self.terminated_torch = wp.to_torch(self.terminated)
        self.time_out_torch = wp.to_torch(self.time_out)
        self.reward_torch = wp.to_torch(self.reward)

    def bind_environment_outputs(self, env) -> None:
        """Bind the environment's intermediate attributes to persistent zero-copy tensor views."""
        env.up_proj = self.up_projection_torch
        env.heading_proj = self.heading_projection_torch
        env.up_vec = self.up_vector_torch
        env.heading_vec = self.heading_vector_torch
        env.vel_loc = self.local_velocity_torch
        env.angvel_loc = self.local_angular_velocity_torch
        env.roll = self.roll_torch
        env.pitch = self.pitch_torch
        env.yaw = self.yaw_torch
        env.angle_to_target = self.angle_to_target_torch
        env.dof_pos_scaled = self.scaled_joint_position_torch

    def compute_post_step(self, env) -> None:
        """Compute locomotion intermediates, dones, reward, and observations in one launch."""
        self._observation_index ^= 1
        self.observation = self._observations[self._observation_index]
        self.observation_torch = self._observation_torch[self._observation_index]
        wp.launch(
            compute_locomotion_post_step,
            dim=self.num_envs,
            inputs=[
                env.targets,
                env.torso_position,
                env.torso_rotation,
                env.velocity,
                env.ang_velocity,
                env.dof_pos,
                env.dof_vel,
                env._joint_position_limits,
                env.actions,
                env.episode_length_buf,
                env.motor_effort_ratio,
                env.cfg.sim.dt,
                env.cfg.angular_velocity_scale,
                env.cfg.dof_vel_scale,
                env.cfg.termination_height,
                env.max_episode_length,
                env.cfg.up_weight,
                env.cfg.heading_weight,
                env.cfg.actions_cost_scale,
                env.cfg.energy_cost_scale,
                env.cfg.death_cost,
                env.cfg.alive_reward_scale,
                self.num_joints,
                env.potentials,
                env.prev_potentials,
                self.up_projection,
                self.heading_projection,
                self.up_vector,
                self.heading_vector,
                self.local_velocity,
                self.local_angular_velocity,
                self.roll,
                self.pitch,
                self.yaw,
                self.angle_to_target,
                self.scaled_joint_position,
                self.terminated,
                self.time_out,
                self.reward,
                self.observation,
            ],
            device=self.device,
        )

    def compute_intermediate_and_observation(self, env) -> None:
        """Refresh intermediates and observations after reset without changing reward or dones."""
        wp.launch(
            compute_locomotion_intermediate_and_observation,
            dim=self.num_envs,
            inputs=[
                env.targets,
                env.torso_position,
                env.torso_rotation,
                env.velocity,
                env.ang_velocity,
                env.dof_pos,
                env.dof_vel,
                env._joint_position_limits,
                env.actions,
                env.cfg.sim.dt,
                env.cfg.angular_velocity_scale,
                env.cfg.dof_vel_scale,
                self.num_joints,
                env.potentials,
                env.prev_potentials,
                self.up_projection,
                self.heading_projection,
                self.up_vector,
                self.heading_vector,
                self.local_velocity,
                self.local_angular_velocity,
                self.roll,
                self.pitch,
                self.yaw,
                self.angle_to_target,
                self.scaled_joint_position,
                self.observation,
            ],
            device=self.device,
        )

    def compute_masked_reset_observation(self, env, env_mask: wp.array(dtype=wp.bool)) -> None:
        """Refresh reset rows and zero their episode lengths without changing terminal outputs."""
        wp.launch(
            compute_locomotion_masked_reset_observation,
            dim=self.num_envs,
            inputs=[
                env_mask,
                env.episode_length_buf,
                env.targets,
                env.torso_position,
                env.torso_rotation,
                env.velocity,
                env.ang_velocity,
                env.dof_pos,
                env.dof_vel,
                env._joint_position_limits,
                env.actions,
                env.cfg.sim.dt,
                env.cfg.angular_velocity_scale,
                env.cfg.dof_vel_scale,
                self.num_joints,
                env.potentials,
                env.prev_potentials,
                self.up_projection,
                self.heading_projection,
                self.up_vector,
                self.heading_vector,
                self.local_velocity,
                self.local_angular_velocity,
                self.roll,
                self.pitch,
                self.yaw,
                self.angle_to_target,
                self.scaled_joint_position,
                self.observation,
            ],
            device=self.device,
        )
