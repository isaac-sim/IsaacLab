# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import warp as wp
from isaaclab_experimental.envs import DirectRLEnvWarp

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnvCfg


@wp.func
def fmod(x: wp.float32, y: wp.float32) -> wp.float32:
    return x - y * wp.floor(x / y)


@wp.func
def euler_from_quat(q: wp.quatf) -> wp.vec3f:
    sinr_cosp = 2.0 * (q[3] * q[0] + q[1] * q[2])
    cosr_cosp = q[3] * q[3] - q[0] * q[0] - q[1] * q[1] + q[2] * q[2]
    sinp = 2.0 * (q[3] * q[1] - q[2] * q[0])
    siny_cosp = 2.0 * (q[3] * q[2] + q[0] * q[1])
    cosy_cosp = q[3] * q[3] + q[0] * q[0] - q[1] * q[1] - q[2] * q[2]
    roll = wp.atan2(sinr_cosp, cosr_cosp)
    if wp.abs(sinp) >= 1:
        pitch = wp.sign(sinp) * wp.pi / 2.0
    else:
        pitch = wp.asin(sinp)
    yaw = wp.atan2(siny_cosp, cosy_cosp)

    return wp.vec3f(
        fmod(roll, wp.static(2.0 * wp.pi)),
        fmod(pitch, wp.static(2.0 * wp.pi)),
        fmod(yaw, wp.static(2.0 * wp.pi)),
    )


@wp.kernel
def get_dones(
    episode_length_buf: wp.array(dtype=wp.int32),
    torso_pose: wp.array(dtype=wp.transformf),
    max_episode_length: wp.int32,
    termination_height: wp.float32,
    out_of_bounds: wp.array(dtype=wp.bool),
    time_out: wp.array(dtype=wp.bool),
    reset: wp.array(dtype=wp.bool),
):
    env_index = wp.tid()
    out_of_bounds[env_index] = wp.abs(torso_pose[env_index][2]) < termination_height
    time_out[env_index] = episode_length_buf[env_index] >= (max_episode_length - 1)
    reset[env_index] = out_of_bounds[env_index] or time_out[env_index]


@wp.kernel
def observations(
    torso_pose: wp.array(dtype=wp.transformf),
    velocity: wp.array(dtype=wp.spatial_vectorf),
    rpy: wp.array(dtype=wp.vec3f),
    angle_to_target: wp.array(dtype=wp.float32),
    up_proj: wp.array(dtype=wp.float32),
    heading_proj: wp.array(dtype=wp.float32),
    dof_pos_scaled: wp.array2d(dtype=wp.float32),
    dof_vel: wp.array2d(dtype=wp.float32),
    actions: wp.array2d(dtype=wp.float32),
    observations: wp.array2d(dtype=wp.float32),
    dof_vel_scale: wp.float32,
    angular_velocity_scale: wp.float32,
    num_dof: wp.int32,
):
    env_index = wp.tid()
    observations[env_index, 0] = torso_pose[env_index][2]
    observations[env_index, 1] = velocity[env_index][0]
    observations[env_index, 2] = velocity[env_index][1]
    observations[env_index, 3] = velocity[env_index][2]
    observations[env_index, 4] = velocity[env_index][3] * angular_velocity_scale
    observations[env_index, 5] = velocity[env_index][4] * angular_velocity_scale
    observations[env_index, 6] = velocity[env_index][5] * angular_velocity_scale
    observations[env_index, 7] = wp.atan2(wp.sin(rpy[env_index][2]), wp.cos(rpy[env_index][2]))
    observations[env_index, 8] = wp.atan2(wp.sin(rpy[env_index][0]), wp.cos(rpy[env_index][0]))
    observations[env_index, 9] = wp.atan2(wp.sin(angle_to_target[env_index]), wp.cos(angle_to_target[env_index]))
    observations[env_index, 10] = up_proj[env_index]
    observations[env_index, 11] = heading_proj[env_index]

    offset_1 = 12 + num_dof
    offset_2 = offset_1 + num_dof

    for i in range(num_dof):
        observations[env_index, 12 + i] = dof_pos_scaled[env_index, i]
    for i in range(num_dof):
        observations[env_index, offset_1 + i] = dof_vel[env_index, i] * dof_vel_scale
    for i in range(num_dof):
        observations[env_index, offset_2 + i] = actions[env_index, i]


@wp.func
def translate_transform(
    transform: wp.transformf,
    translation: wp.vec3f,
) -> wp.transformf:
    return wp.transform(
        wp.transform_get_translation(transform) + translation,
        wp.transform_get_rotation(transform),
    )


@wp.kernel
def reset_root(
    default_root_pose: wp.array(dtype=wp.transformf),
    default_root_vel: wp.array(dtype=wp.spatial_vectorf),
    env_origins: wp.array(dtype=wp.vec3f),
    dt: wp.float32,
    to_targets: wp.array(dtype=wp.vec3f),
    potentials: wp.array(dtype=wp.float32),
    root_pose: wp.array(dtype=wp.transformf),
    root_vel: wp.array(dtype=wp.spatial_vectorf),
    env_mask: wp.array(dtype=wp.bool),
):
    env_index = wp.tid()
    if env_mask[env_index]:
        root_pose[env_index] = default_root_pose[env_index]
        root_pose[env_index] = translate_transform(root_pose[env_index], env_origins[env_index])
        root_vel[env_index] = default_root_vel[env_index]
        to_targets[env_index] = wp.transform_get_translation(root_pose[env_index]) - wp.transform_get_translation(
            default_root_pose[env_index]
        )
        to_targets[env_index][2] = 0.0
        potentials[env_index] = -wp.length(to_targets[env_index]) / dt


@wp.kernel
def reset_joints(
    default_joint_pos: wp.array2d(dtype=wp.float32),
    default_joint_vel: wp.array2d(dtype=wp.float32),
    joint_pos: wp.array2d(dtype=wp.float32),
    joint_vel: wp.array2d(dtype=wp.float32),
    env_mask: wp.array(dtype=wp.bool),
):
    env_index, joint_index = wp.tid()
    if env_mask[env_index]:
        joint_pos[env_index, joint_index] = default_joint_pos[env_index, joint_index]
        joint_vel[env_index, joint_index] = default_joint_vel[env_index, joint_index]


@wp.func
def heading_reward(
    heading_proj: wp.float32,
    heading_weight: wp.float32,
) -> wp.float32:
    if heading_proj > 0.8:
        return heading_weight
    else:
        return heading_weight * heading_proj / 0.8


@wp.func
def up_reward(
    up_proj: wp.float32,
    up_weight: wp.float32,
) -> wp.float32:

    if up_proj > 0.93:
        return up_weight
    else:
        return 0.0


@wp.func
def progress_reward(
    current_value: wp.float32,
    prev_value: wp.float32,
) -> wp.float32:
    return current_value - prev_value


@wp.func
def actions_cost(
    actions: wp.array(dtype=wp.float32),
) -> wp.float32:
    sum_ = wp.float32(0.0)
    for i in range(len(actions)):
        sum_ += actions[i] * actions[i]
    return sum_


@wp.func
def electricity_cost(
    actions: wp.array(dtype=wp.float32),
    dof_vel: wp.array(dtype=wp.float32),
    dof_vel_scale: wp.float32,
    motor_effort_ratio: wp.array(dtype=wp.float32),
) -> wp.float32:
    sum_ = wp.float32(0.0)
    for i in range(len(actions)):
        sum_ += wp.abs(actions[i] * dof_vel[i] * dof_vel_scale) * motor_effort_ratio[i]
    return sum_


@wp.func
def dof_at_limit_cost(
    dof_pos_scaled: wp.array(dtype=wp.float32),
) -> wp.float32:
    sum_ = wp.float32(0.0)
    for i in range(len(dof_pos_scaled)):
        if dof_pos_scaled[i] > 0.98:
            sum_ += 1.0
    return sum_


@wp.kernel
def compute_rewards(
    actions: wp.array2d(dtype=wp.float32),
    dof_vel: wp.array2d(dtype=wp.float32),
    dof_pos_scaled: wp.array2d(dtype=wp.float32),
    reset_terminated: wp.array(dtype=wp.bool),
    heading_proj: wp.array(dtype=wp.float32),
    up_proj: wp.array(dtype=wp.float32),
    potentials: wp.array(dtype=wp.float32),
    prev_potentials: wp.array(dtype=wp.float32),
    motor_effort_ratio: wp.array(dtype=wp.float32),
    up_weight: wp.float32,
    heading_weight: wp.float32,
    actions_cost_scale: wp.float32,
    energy_cost_scale: wp.float32,
    dof_vel_scale: wp.float32,
    death_cost: wp.float32,
    alive_reward_scale: wp.float32,
    reward: wp.array(dtype=wp.float32),
):
    env_index = wp.tid()
    if reset_terminated[env_index]:
        reward[env_index] = death_cost
    else:
        reward[env_index] = (
            progress_reward(potentials[env_index], prev_potentials[env_index])
            + alive_reward_scale
            + up_reward(up_proj[env_index], up_weight)
            + heading_reward(heading_proj[env_index], heading_weight)
            - actions_cost_scale * actions_cost(actions[env_index])
            - energy_cost_scale
            * electricity_cost(actions[env_index], dof_vel[env_index], dof_vel_scale, motor_effort_ratio)
            - dof_at_limit_cost(dof_pos_scaled[env_index])
        )


@wp.kernel
def compute_heading_and_up(
    torso_pose: wp.array(dtype=wp.transformf),
    targets: wp.array(dtype=wp.vec3f),
    dt: wp.float32,
    to_targets: wp.array(dtype=wp.vec3f),
    up_proj: wp.array(dtype=wp.float32),
    heading_proj: wp.array(dtype=wp.float32),
    up_vec: wp.array(dtype=wp.vec3f),
    heading_vec: wp.array(dtype=wp.vec3f),
    potentials: wp.array(dtype=wp.float32),
    prev_potentials: wp.array(dtype=wp.float32),
):
    env_index = wp.tid()
    up_vec[env_index] = wp.quat_rotate(wp.transform_get_rotation(torso_pose[env_index]), wp.static(wp.vec3f(0, 0, 1)))
    heading_vec[env_index] = wp.quat_rotate(
        wp.transform_get_rotation(torso_pose[env_index]), wp.static(wp.vec3f(1, 0, 0))
    )
    up_proj[env_index] = up_vec[env_index][2]
    to_targets[env_index] = targets[env_index] - wp.transform_get_translation(torso_pose[env_index])
    to_targets[env_index][2] = 0.0
    heading_proj[env_index] = wp.dot(heading_vec[env_index], wp.normalize(to_targets[env_index]))
    prev_potentials[env_index] = potentials[env_index]
    potentials[env_index] = -wp.length(to_targets[env_index]) / dt


@wp.func
def spatial_rotate_inv(quat: wp.quatf, vec: wp.spatial_vectorf) -> wp.spatial_vectorf:
    return wp.spatial_vector(
        wp.quat_rotate_inv(quat, wp.spatial_top(vec)),
        wp.quat_rotate_inv(quat, wp.spatial_bottom(vec)),
    )


@wp.func
def unscale(x: wp.float32, lower: wp.float32, upper: wp.float32) -> wp.float32:
    return (2.0 * x - upper - lower) / (upper - lower)


@wp.kernel
def compute_rot(
    torso_pose: wp.array(dtype=wp.transformf),
    velocity: wp.array(dtype=wp.spatial_vectorf),
    targets: wp.array(dtype=wp.vec3f),
    vec_loc: wp.array(dtype=wp.spatial_vectorf),
    rpy: wp.array(dtype=wp.vec3f),
    angle_to_target: wp.array(dtype=wp.float32),
):
    env_index = wp.tid()
    vec_loc[env_index] = spatial_rotate_inv(wp.transform_get_rotation(torso_pose[env_index]), velocity[env_index])
    rpy[env_index] = euler_from_quat(wp.transform_get_rotation(torso_pose[env_index]))
    angle_to_target[env_index] = (
        wp.atan2(targets[env_index][1] - torso_pose[env_index][1], targets[env_index][0] - torso_pose[env_index][0])
        - rpy[env_index][2]
    )


@wp.kernel
def scale_dof_pos(
    dof_pos: wp.array2d(dtype=wp.float32),
    dof_limits: wp.array2d(dtype=wp.vec2f),
    dof_pos_scaled: wp.array2d(dtype=wp.float32),
):
    env_index, joint_index = wp.tid()
    dof_pos_scaled[env_index, joint_index] = unscale(
        dof_pos[env_index, joint_index], dof_limits[env_index, joint_index][0], dof_limits[env_index, joint_index][1]
    )


@wp.kernel
def update_actions(
    input_actions: wp.array2d(dtype=wp.float32),
    actions: wp.array2d(dtype=wp.float32),
    joint_gears: wp.array(dtype=wp.float32),
    action_scale: wp.float32,
):
    env_index, joint_index = wp.tid()
    actions[env_index, joint_index] = (
        action_scale * joint_gears[joint_index] * wp.clamp(input_actions[env_index, joint_index], -1.0, 1.0)
    )


@wp.kernel
def initialize_state(
    env_origins: wp.array(dtype=wp.vec3f),
    targets: wp.array(dtype=wp.vec3f),
    state: wp.array(dtype=wp.uint32),
    seed: wp.int32,
):
    env_index = wp.tid()
    state[env_index] = wp.rand_init(seed, env_index)
    targets[env_index] = env_origins[env_index]
    targets[env_index] += wp.static(wp.vec3f(1000.0, 0.0, 0.0))


class LocomotionWarpEnv(DirectRLEnvWarp):
    cfg: DirectRLEnvCfg

    def __init__(self, cfg: DirectRLEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.action_scale = self.cfg.action_scale
        self.joint_gears = wp.array(self.cfg.joint_gears, dtype=wp.float32, device=self.sim.device)
        self.motor_effort_ratio = wp.ones_like(self.joint_gears, device=self.sim.device)
        self._joint_dof_mask, _, self._joint_dof_idx = self.robot.find_joints(".*")

        # Simulation bindings
        # Note: these are direct memory views into the Newton simulation data, they should not be modified directly
        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel
        self.root_pose_w = self.robot.data.root_pose_w
        self.root_vel_w = self.robot.data.root_vel_w
        self.soft_joint_pos_limits = self.robot.data.soft_joint_pos_limits

        # Buffers
        self.observations = wp.zeros(
            (self.num_envs, self.cfg.observation_space), dtype=wp.float32, device=self.sim.device
        )
        self.rewards = wp.zeros((self.num_envs), dtype=wp.float32, device=self.sim.device)
        self.actions = wp.zeros((self.num_envs, self.robot.num_joints), dtype=wp.float32, device=self.sim.device)
        self.states = wp.zeros((self.num_envs), dtype=wp.uint32, device=self.sim.device)
        self.potentials = wp.zeros(self.num_envs, dtype=wp.float32, device=self.sim.device)
        self.prev_potentials = wp.zeros_like(self.potentials)
        self.targets = wp.zeros((self.num_envs), dtype=wp.vec3f, device=self.sim.device)
        self.up_vec = wp.zeros((self.num_envs), dtype=wp.vec3f, device=self.sim.device)
        self.heading_vec = wp.zeros((self.num_envs), dtype=wp.vec3f, device=self.sim.device)
        self.to_targets = wp.zeros((self.num_envs), dtype=wp.vec3f, device=self.sim.device)
        self.up_proj = wp.zeros((self.num_envs), dtype=wp.float32, device=self.sim.device)
        self.heading_proj = wp.zeros((self.num_envs), dtype=wp.float32, device=self.sim.device)
        self.vec_loc = wp.zeros((self.num_envs), dtype=wp.spatial_vectorf, device=self.sim.device)
        self.rpy = wp.zeros((self.num_envs), dtype=wp.vec3f, device=self.sim.device)
        self.angle_to_target = wp.zeros((self.num_envs), dtype=wp.float32, device=self.sim.device)
        self.dof_pos_scaled = wp.zeros((self.num_envs, self.robot.num_joints), dtype=wp.float32, device=self.sim.device)
        self.env_origins = wp.from_torch(self.scene.env_origins, dtype=wp.vec3f)
        self.actions_mapped = wp.zeros((self.num_envs, self.robot.num_joints), dtype=wp.float32, device=self.sim.device)

        # Initial states and targets
        if self.cfg.seed is None:
            self.cfg.seed = -1

        wp.launch(
            initialize_state,
            dim=self.num_envs,
            inputs=[
                self.env_origins,
                self.targets,
                self.states,
                self.cfg.seed,
            ],
        )

        # Bind torch buffers to warp buffers
        self.torch_obs_buf = wp.to_torch(self.observations)
        self.torch_reward_buf = wp.to_torch(self.rewards)
        self.torch_reset_terminated = wp.to_torch(self.reset_terminated)
        self.torch_reset_time_outs = wp.to_torch(self.reset_time_outs)
        self.torch_episode_length_buf = wp.to_torch(self.episode_length_buf)

    def _setup_scene(self) -> None:
        self.robot = Articulation(self.cfg.robot)
        # add ground plane
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self.terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # add articulation to scene
        self.scene.articulations["robot"] = self.robot
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: wp.array) -> None:
        self.actions.assign(actions)
        wp.launch(
            update_actions,
            dim=(self.num_envs, self.robot.num_joints),
            inputs=[actions, self.actions_mapped, self.joint_gears, self.action_scale],
        )

    def _apply_action(self) -> None:
        self.robot.set_joint_effort_target(self.actions_mapped, joint_mask=self._joint_dof_mask)

    def _compute_intermediate_values(self) -> None:
        wp.launch(
            compute_heading_and_up,
            dim=self.num_envs,
            inputs=[
                self.root_pose_w,
                self.targets,
                self.cfg.sim.dt,
                self.to_targets,
                self.up_proj,
                self.heading_proj,
                self.up_vec,
                self.heading_vec,
                self.potentials,
                self.prev_potentials,
            ],
        )

        wp.launch(
            compute_rot,
            dim=self.num_envs,
            inputs=[
                self.root_pose_w,
                self.root_vel_w,
                self.targets,
                self.vec_loc,
                self.rpy,
                self.angle_to_target,
            ],
        )
        wp.launch(
            scale_dof_pos,
            dim=(self.num_envs, self.robot.num_joints),
            inputs=[
                self.joint_pos,
                self.soft_joint_pos_limits,
                self.dof_pos_scaled,
            ],
        )

    def _get_observations(self) -> None:
        wp.launch(
            observations,
            dim=self.num_envs,
            inputs=[
                self.root_pose_w,
                self.vec_loc,
                self.rpy,
                self.angle_to_target,
                self.up_proj,
                self.heading_proj,
                self.dof_pos_scaled,
                self.joint_vel,
                self.actions,
                self.observations,
                self.cfg.dof_vel_scale,
                self.cfg.angular_velocity_scale,
                self.robot.num_joints,
            ],
        )

    def _get_rewards(self) -> None:
        wp.launch(
            compute_rewards,
            dim=self.num_envs,
            inputs=[
                self.actions,
                self.joint_vel,
                self.dof_pos_scaled,
                self.reset_terminated,
                self.heading_proj,
                self.up_proj,
                self.potentials,
                self.prev_potentials,
                self.motor_effort_ratio,
                self.cfg.up_weight,
                self.cfg.heading_weight,
                self.cfg.actions_cost_scale,
                self.cfg.energy_cost_scale,
                self.cfg.dof_vel_scale,
                self.cfg.death_cost,
                self.cfg.alive_reward_scale,
                self.rewards,
            ],
        )

    def _get_dones(self) -> None:
        self._compute_intermediate_values()

        wp.launch(
            get_dones,
            dim=self.num_envs,
            inputs=[
                self.episode_length_buf,
                self.root_pose_w,
                self.max_episode_length,
                self.cfg.termination_height,
                self.reset_terminated,
                self.reset_time_outs,
                self.reset_buf,
            ],
        )

    def _reset_idx(self, mask: wp.array | None = None):
        if mask is None:
            mask = self.robot._ALL_ENV_MASK

        super()._reset_idx(mask)

        wp.launch(
            reset_root,
            dim=self.num_envs,
            inputs=[
                self.robot.data.default_root_pose,
                self.robot.data.default_root_vel,
                self.env_origins,
                self.cfg.sim.dt,
                self.to_targets,
                self.potentials,
                self.root_pose_w,
                self.root_vel_w,
                mask,
            ],
        )
        wp.launch(
            reset_joints,
            dim=(self.num_envs, self.robot.num_joints),
            inputs=[
                self.robot.data.default_joint_pos,
                self.robot.data.default_joint_vel,
                self.joint_pos,
                self.joint_vel,
                mask,
            ],
        )

        self._compute_intermediate_values()
