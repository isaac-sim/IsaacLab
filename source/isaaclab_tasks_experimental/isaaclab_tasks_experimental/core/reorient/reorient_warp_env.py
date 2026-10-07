# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import warp as wp
from isaaclab_experimental.envs import DirectRLEnvWarp
from isaaclab_experimental.utils.warp_capture import captured

from isaaclab.utils.seed import WarpRng

if TYPE_CHECKING:
    from isaaclab_tasks.core.reorient.config.allegro_hand.allegro_hand_direct_env_cfg import AllegroHandEnvCfg


@wp.kernel
def apply_actions_to_targets(
    # input
    actions: wp.array2d(dtype=wp.float32),
    lower_limits: wp.array2d(dtype=wp.float32),
    upper_limits: wp.array2d(dtype=wp.float32),
    actuated_dof_indices: wp.array(dtype=wp.int32),
    act_moving_average: wp.float32,
    # input/output
    prev_targets: wp.array2d(dtype=wp.float32),
    # output
    cur_targets: wp.array2d(dtype=wp.float32),
):
    """Scale clamped actions to each joint's range with moving-average smoothing; write current and previous targets."""
    env_id, i = wp.tid()
    dof_id = actuated_dof_indices[i]

    # clamp and scale action to target range
    a = wp.clamp(actions[env_id, i], wp.float32(-1.0), wp.float32(1.0))
    lower = lower_limits[env_id, dof_id]
    upper = upper_limits[env_id, dof_id]
    t = scale(a, lower, upper)

    # smoothing and boundary clamping
    t = act_moving_average * t + (wp.float32(1.0) - act_moving_average) * prev_targets[env_id, dof_id]
    t = wp.clamp(t, lower, upper)

    # update targets
    cur_targets[env_id, dof_id] = t
    prev_targets[env_id, dof_id] = t


@wp.kernel
def reset_target_pose(
    # input
    env_mask: wp.array(dtype=wp.bool),
    x_unit_vec: wp.vec3f,
    y_unit_vec: wp.vec3f,
    env_origins: wp.array(dtype=wp.vec3f),
    goal_pos: wp.array(dtype=wp.vec3f),
    # input/output
    rng_state: wp.array(dtype=wp.uint32),
    # output
    goal_rot: wp.array(dtype=wp.quatf),
    reset_goal_buf: wp.array(dtype=wp.bool),
    goal_pos_w: wp.array(dtype=wp.vec3f),
):
    """Sample a randomized goal orientation and compute the world-frame goal position for masked envs."""
    env_id = wp.tid()
    if env_mask[env_id]:
        rand0 = wp.randf(rng_state[env_id], wp.float32(-1.0), wp.float32(1.0))
        rng_state[env_id] += wp.uint32(1)
        rand1 = wp.randf(rng_state[env_id], wp.float32(-1.0), wp.float32(1.0))
        rng_state[env_id] += wp.uint32(1)

        goal_rot[env_id] = randomize_rotation(rand0, rand1, x_unit_vec, y_unit_vec)
        reset_goal_buf[env_id] = False

        # Warp-native addition: goal position in world frame.
        goal_pos_w[env_id] = goal_pos[env_id] + env_origins[env_id]


@wp.kernel
def reset_object(
    # input
    default_root_pose: wp.array(dtype=wp.transformf),
    env_origins: wp.array(dtype=wp.vec3f),
    reset_position_noise: wp.float32,
    x_unit_vec: wp.vec3f,
    y_unit_vec: wp.vec3f,
    env_mask: wp.array(dtype=wp.bool),
    # input/output
    rng_state: wp.array(dtype=wp.uint32),
    # output
    root_pose_w: wp.array(dtype=wp.transformf),
    root_vel_w: wp.array(dtype=wp.spatial_vectorf),
):
    """Sample masked envs' object root pose with position noise and a randomized rotation, and a zero velocity."""
    env_id = wp.tid()
    if env_mask[env_id]:
        nx = wp.randf(rng_state[env_id], wp.float32(-1.0), wp.float32(1.0))
        rng_state[env_id] += wp.uint32(1)
        ny = wp.randf(rng_state[env_id], wp.float32(-1.0), wp.float32(1.0))
        rng_state[env_id] += wp.uint32(1)
        nz = wp.randf(rng_state[env_id], wp.float32(-1.0), wp.float32(1.0))
        rng_state[env_id] += wp.uint32(1)

        pos_noise = reset_position_noise * wp.vec3f(nx, ny, nz)
        base_pos = wp.transform_get_translation(default_root_pose[env_id])
        pos_w = base_pos + env_origins[env_id] + pos_noise

        rand0 = wp.randf(rng_state[env_id], wp.float32(-1.0), wp.float32(1.0))
        rng_state[env_id] += wp.uint32(1)
        rand1 = wp.randf(rng_state[env_id], wp.float32(-1.0), wp.float32(1.0))
        rng_state[env_id] += wp.uint32(1)
        rot_w = randomize_rotation(rand0, rand1, x_unit_vec, y_unit_vec)

        root_pose_w[env_id] = wp.transform(pos_w, rot_w)
        root_vel_w[env_id] = wp.spatial_vectorf(
            wp.float32(0.0), wp.float32(0.0), wp.float32(0.0), wp.float32(0.0), wp.float32(0.0), wp.float32(0.0)
        )


@wp.kernel
def reset_hand(
    # input
    default_joint_pos: wp.array2d(dtype=wp.float32),
    default_joint_vel: wp.array2d(dtype=wp.float32),
    lower_limits: wp.array2d(dtype=wp.float32),
    upper_limits: wp.array2d(dtype=wp.float32),
    reset_dof_pos_noise: wp.float32,
    reset_dof_vel_noise: wp.float32,
    env_mask: wp.array(dtype=wp.bool),
    num_dofs: wp.int32,
    # input/output
    rng_state: wp.array(dtype=wp.uint32),
    # output
    joint_pos: wp.array2d(dtype=wp.float32),
    joint_vel: wp.array2d(dtype=wp.float32),
    prev_targets: wp.array2d(dtype=wp.float32),
    cur_targets: wp.array2d(dtype=wp.float32),
):
    """Sample masked envs' reset joint positions and velocities with noise and initialize their position targets."""
    env_id = wp.tid()
    if env_mask[env_id]:
        # Each env runs sequentially inside this kernel (avoids RNG races across DOFs).
        for dof_id in range(num_dofs):
            dof_pos_noise = wp.randf(rng_state[env_id], wp.float32(-1.0), wp.float32(1.0))
            rng_state[env_id] += wp.uint32(1)

            delta_max = upper_limits[env_id, dof_id] - default_joint_pos[env_id, dof_id]
            delta_min = lower_limits[env_id, dof_id] - default_joint_pos[env_id, dof_id]
            rand_delta = delta_min + (delta_max - delta_min) * 0.5 * (dof_pos_noise + wp.float32(1.0))
            pos = default_joint_pos[env_id, dof_id] + reset_dof_pos_noise * rand_delta

            dof_vel_noise = wp.randf(rng_state[env_id], wp.float32(-1.0), wp.float32(1.0))
            rng_state[env_id] += wp.uint32(1)
            vel = default_joint_vel[env_id, dof_id] + reset_dof_vel_noise * dof_vel_noise

            joint_pos[env_id, dof_id] = pos
            joint_vel[env_id, dof_id] = vel
            prev_targets[env_id, dof_id] = pos
            cur_targets[env_id, dof_id] = pos


@wp.kernel
def reset_successes(
    # input
    env_mask: wp.array(dtype=wp.bool),
    # input/output
    successes: wp.array(dtype=wp.float32),
    success_rate_stats: wp.array(dtype=wp.float32),
):
    """Accumulate masked envs' finished-episode success rate into ``success_rate_stats``, then zero their counts.

    Reaching a goal draws a replacement, so an episode that reached ``n`` goals attempted ``n + 1``.
    ``success_rate_stats`` must be zeroed before the launch; ``[0]`` receives the summed per-episode
    rates and ``[1]`` the number of finished episodes.
    """
    env_id = wp.tid()
    if env_mask[env_id]:
        goals = successes[env_id]
        wp.atomic_add(success_rate_stats, 0, goals / (goals + wp.float32(1.0)))
        wp.atomic_add(success_rate_stats, 1, wp.float32(1.0))
        successes[env_id] = wp.float32(0.0)


@wp.kernel
def update_success_rate_from_stats(
    # input
    success_rate_stats: wp.array(dtype=wp.float32),
    # output
    success_rate: wp.array(dtype=wp.float32),
):
    """Average the accumulated episode success rates; keep the last value when no episode finished."""
    # single-thread kernel (dim=1)
    if success_rate_stats[1] > wp.float32(0.0):
        success_rate[0] = success_rate_stats[0] / success_rate_stats[1]


@wp.kernel
def compute_intermediate_values(
    # input
    body_pos_w: wp.array2d(dtype=wp.vec3f),
    body_quat_w: wp.array2d(dtype=wp.quatf),
    body_vel_w: wp.array2d(dtype=wp.spatial_vectorf),
    finger_bodies: wp.array(dtype=wp.int32),
    env_origins: wp.array(dtype=wp.vec3f),
    object_root_pose_w: wp.array(dtype=wp.transformf),
    object_root_vel_w: wp.array(dtype=wp.spatial_vectorf),
    num_fingertips: wp.int32,
    # output
    fingertip_pos: wp.array2d(dtype=wp.vec3f),
    fingertip_rot: wp.array2d(dtype=wp.quatf),
    fingertip_velocities: wp.array2d(dtype=wp.spatial_vectorf),
    object_pose: wp.array(dtype=wp.transformf),
    object_vels: wp.array(dtype=wp.spatial_vectorf),
):
    """Compute per-env fingertip poses/velocities and the env-local object pose and velocity."""
    env_id = wp.tid()

    for i in range(num_fingertips):
        body_id = finger_bodies[i]
        fingertip_pos[env_id, i] = body_pos_w[env_id, body_id] - env_origins[env_id]
        fingertip_rot[env_id, i] = body_quat_w[env_id, body_id]
        fingertip_velocities[env_id, i] = body_vel_w[env_id, body_id]

    # Store object pose in env-local frame (translation only; orientation unchanged).
    pos_w = wp.transform_get_translation(object_root_pose_w[env_id])
    pos = pos_w - env_origins[env_id]
    rot = wp.transform_get_rotation(object_root_pose_w[env_id])
    object_pose[env_id] = wp.transform(pos, rot)
    object_vels[env_id] = object_root_vel_w[env_id]


@wp.kernel
def get_dones(
    # input
    max_episode_length: wp.int32,
    object_pose: wp.array(dtype=wp.transformf),
    in_hand_pos: wp.array(dtype=wp.vec3f),
    goal_rot: wp.array(dtype=wp.quatf),
    fall_dist: wp.float32,
    success_tolerance: wp.float32,
    max_consecutive_success: wp.int32,
    successes: wp.array(dtype=wp.float32),
    # input/output
    episode_length_buf: wp.array(dtype=wp.int32),
    # output
    orientation_error: wp.array(dtype=wp.float32),
    goal_reached: wp.array(dtype=wp.bool),
    out_of_reach: wp.array(dtype=wp.bool),
    time_out: wp.array(dtype=wp.bool),
    reset: wp.array(dtype=wp.bool),
):
    """Evaluate goal success once per step, flag object-fall and time-out/max-success termination.

    The orientation error [rad] and success flags are written for the reward to reuse. Progress is
    reset on reaching a goal when a consecutive-success cap is configured.
    """
    env_id = wp.tid()

    object_pos = wp.transform_get_translation(object_pose[env_id])
    object_rot = wp.transform_get_rotation(object_pose[env_id])

    goal_dist = wp.length(object_pos - in_hand_pos[env_id])
    out_of_reach[env_id] = goal_dist >= fall_dist

    error = rotation_distance(object_rot, goal_rot[env_id])
    reached = error <= success_tolerance
    orientation_error[env_id] = error
    goal_reached[env_id] = reached

    max_success_reached = False
    if max_consecutive_success > 0:
        if reached:
            episode_length_buf[env_id] = 0
        max_success_reached = successes[env_id] >= wp.float32(max_consecutive_success)

    time_out[env_id] = episode_length_buf[env_id] >= (max_episode_length - 1) or max_success_reached
    reset[env_id] = out_of_reach[env_id] or time_out[env_id]


@wp.kernel
def compute_reduced_observations(
    # input
    fingertip_pos: wp.array2d(dtype=wp.vec3f),
    object_pose: wp.array(dtype=wp.transformf),
    goal_rot: wp.array(dtype=wp.quatf),
    actions: wp.array2d(dtype=wp.float32),
    num_fingertips: wp.int32,
    action_dim: wp.int32,
    # output
    observations: wp.array2d(dtype=wp.float32),
):
    """Assemble the reduced observation vector (fingertip/object positions, relative goal orientation, actions)."""
    env_id = wp.tid()

    obj_pos = wp.transform_get_translation(object_pose[env_id])
    obj_rot = wp.transform_get_rotation(object_pose[env_id])

    idx = int(0)
    for i in range(num_fingertips):
        observations[env_id, idx + 0] = fingertip_pos[env_id, i][0]
        observations[env_id, idx + 1] = fingertip_pos[env_id, i][1]
        observations[env_id, idx + 2] = fingertip_pos[env_id, i][2]
        idx += 3

    observations[env_id, idx + 0] = obj_pos[0]
    observations[env_id, idx + 1] = obj_pos[1]
    observations[env_id, idx + 2] = obj_pos[2]
    idx += 3

    rel = obj_rot * wp.quat_inverse(goal_rot[env_id])
    observations[env_id, idx + 0] = rel[0]
    observations[env_id, idx + 1] = rel[1]
    observations[env_id, idx + 2] = rel[2]
    observations[env_id, idx + 3] = rel[3]
    idx += 4

    for i in range(action_dim):
        observations[env_id, idx + i] = actions[env_id, i]


@wp.kernel
def compute_full_observations(
    # input
    hand_dof_pos: wp.array2d(dtype=wp.float32),
    hand_dof_vel: wp.array2d(dtype=wp.float32),
    hand_dof_lower_limits: wp.array2d(dtype=wp.float32),
    hand_dof_upper_limits: wp.array2d(dtype=wp.float32),
    vel_obs_scale: wp.float32,
    object_pose: wp.array(dtype=wp.transformf),
    object_vels: wp.array(dtype=wp.spatial_vectorf),
    in_hand_pos: wp.array(dtype=wp.vec3f),
    goal_rot: wp.array(dtype=wp.quatf),
    fingertip_pos: wp.array2d(dtype=wp.vec3f),
    fingertip_rot: wp.array2d(dtype=wp.quatf),
    fingertip_velocities: wp.array2d(dtype=wp.spatial_vectorf),
    actions: wp.array2d(dtype=wp.float32),
    num_hand_dofs: wp.int32,
    num_fingertips: wp.int32,
    action_dim: wp.int32,
    # output
    observations: wp.array2d(dtype=wp.float32),
):
    """Assemble the full observation vector (scaled DOF pos/vel, object pose/vel, goal, fingertip states, actions)."""
    env_id = wp.tid()

    # hand
    for i in range(num_hand_dofs):
        observations[env_id, i] = unscale(
            hand_dof_pos[env_id, i], hand_dof_lower_limits[env_id, i], hand_dof_upper_limits[env_id, i]
        )

    offset = num_hand_dofs
    for i in range(num_hand_dofs):
        observations[env_id, offset + i] = vel_obs_scale * hand_dof_vel[env_id, i]
    offset += num_hand_dofs

    # object
    obj_pos = wp.transform_get_translation(object_pose[env_id])
    obj_rot = wp.transform_get_rotation(object_pose[env_id])

    observations[env_id, offset + 0] = obj_pos[0]
    observations[env_id, offset + 1] = obj_pos[1]
    observations[env_id, offset + 2] = obj_pos[2]
    offset += 3

    observations[env_id, offset + 0] = obj_rot[0]
    observations[env_id, offset + 1] = obj_rot[1]
    observations[env_id, offset + 2] = obj_rot[2]
    observations[env_id, offset + 3] = obj_rot[3]
    offset += 4

    # root velocities are laid out [lin_vel, ang_vel]; only the angular part is scaled
    observations[env_id, offset + 0] = object_vels[env_id][0]
    observations[env_id, offset + 1] = object_vels[env_id][1]
    observations[env_id, offset + 2] = object_vels[env_id][2]
    offset += 3

    observations[env_id, offset + 0] = vel_obs_scale * object_vels[env_id][3]
    observations[env_id, offset + 1] = vel_obs_scale * object_vels[env_id][4]
    observations[env_id, offset + 2] = vel_obs_scale * object_vels[env_id][5]
    offset += 3

    # goal
    observations[env_id, offset + 0] = in_hand_pos[env_id][0]
    observations[env_id, offset + 1] = in_hand_pos[env_id][1]
    observations[env_id, offset + 2] = in_hand_pos[env_id][2]
    offset += 3

    observations[env_id, offset + 0] = goal_rot[env_id][0]
    observations[env_id, offset + 1] = goal_rot[env_id][1]
    observations[env_id, offset + 2] = goal_rot[env_id][2]
    observations[env_id, offset + 3] = goal_rot[env_id][3]
    offset += 4

    rel = obj_rot * wp.quat_inverse(goal_rot[env_id])
    observations[env_id, offset + 0] = rel[0]
    observations[env_id, offset + 1] = rel[1]
    observations[env_id, offset + 2] = rel[2]
    observations[env_id, offset + 3] = rel[3]
    offset += 4

    # fingertips
    for i in range(num_fingertips):
        observations[env_id, offset + 0] = fingertip_pos[env_id, i][0]
        observations[env_id, offset + 1] = fingertip_pos[env_id, i][1]
        observations[env_id, offset + 2] = fingertip_pos[env_id, i][2]
        offset += 3

    for i in range(num_fingertips):
        observations[env_id, offset + 0] = fingertip_rot[env_id, i][0]
        observations[env_id, offset + 1] = fingertip_rot[env_id, i][1]
        observations[env_id, offset + 2] = fingertip_rot[env_id, i][2]
        observations[env_id, offset + 3] = fingertip_rot[env_id, i][3]
        offset += 4

    for i in range(num_fingertips):
        for j in range(6):
            observations[env_id, offset + j] = fingertip_velocities[env_id, i][j]
        offset += 6

    # actions
    for i in range(action_dim):
        observations[env_id, offset + i] = actions[env_id, i]


@wp.kernel
def sanitize_and_print_once(
    # input/output
    obs: wp.array(dtype=wp.float32),
    printed_flag: wp.array(dtype=wp.int32),
):
    """Zero any non-finite observation entry and print a warning once via an atomic print token."""
    i = wp.tid()
    v = obs[i]

    if not wp.isfinite(v):
        # Try to claim the "print token"
        if wp.atomic_cas(printed_flag, 0, 0, 1) == 0:
            wp.printf("Non-finite values in observations")

        obs[i] = wp.float32(0.0)


@wp.kernel
def compute_rewards(
    # input
    reset_buf: wp.array(dtype=wp.bool),
    object_pose: wp.array(dtype=wp.transformf),
    target_pos: wp.array(dtype=wp.vec3f),
    goal_reached: wp.array(dtype=wp.bool),
    orientation_error: wp.array(dtype=wp.float32),
    dist_reward_scale: wp.float32,
    rot_reward_scale: wp.float32,
    rot_eps: wp.float32,
    actions: wp.array2d(dtype=wp.float32),
    action_penalty_scale: wp.float32,
    reach_goal_bonus: wp.float32,
    fall_dist: wp.float32,
    fall_penalty: wp.float32,
    action_dim: wp.int32,
    # input/output
    reset_goal_buf: wp.array(dtype=wp.bool),
    successes: wp.array(dtype=wp.float32),
    num_resets_out: wp.array(dtype=wp.float32),
    finished_cons_successes_out: wp.array(dtype=wp.float32),
    # output
    reward_out: wp.array(dtype=wp.float32),
):
    """Compute the in-hand reorientation reward and update success/reset statistics."""
    env_id = wp.tid()

    obj_pos = wp.transform_get_translation(object_pose[env_id])
    goal_dist = wp.length(obj_pos - target_pos[env_id])

    dist_rew = goal_dist * dist_reward_scale
    rot_rew = rot_reward_scale / (orientation_error[env_id] + rot_eps)

    action_penalty = wp.float32(0.0)
    for i in range(action_dim):
        action_penalty += actions[env_id, i] * actions[env_id, i]

    # Total reward is: position distance + orientation alignment + action regularization + success bonus + fall penalty
    reward = dist_rew + rot_rew + action_penalty * action_penalty_scale

    # a goal stays flagged until it is resampled, which happens later this step
    goal_resets = goal_reached[env_id] or reset_goal_buf[env_id]
    reset_goal_buf[env_id] = goal_resets
    if goal_resets:
        successes[env_id] = successes[env_id] + wp.float32(1.0)
        reward = reward + reach_goal_bonus

    # Fall penalty: distance to the goal is larger than a threshold
    if goal_dist >= fall_dist:
        reward = reward + fall_penalty

    # Consecutive-successes stats (mirrors Torch env):
    #   resets = torch.where(goal_dist >= fall_dist, ones_like(reset_buf), reset_buf)
    resets = (goal_dist >= fall_dist) or reset_buf[env_id]
    if resets:
        wp.atomic_add(num_resets_out, 0, wp.float32(1.0))
        wp.atomic_add(finished_cons_successes_out, 0, successes[env_id])

    reward_out[env_id] = reward


@wp.kernel
def update_consecutive_successes_from_stats(
    # input
    num_resets: wp.array(dtype=wp.float32),
    finished_cons_successes: wp.array(dtype=wp.float32),
    av_factor: wp.float32,
    # input/output
    consecutive_successes: wp.array(dtype=wp.float32),
):
    """Finalize the Torch env's EMA update for consecutive_successes and clear the accumulators."""
    # single-thread kernel (dim=1)
    n = num_resets[0]
    prev = consecutive_successes[0]
    if n > wp.float32(0.0):
        consecutive_successes[0] = av_factor * (finished_cons_successes[0] / n) + (wp.float32(1.0) - av_factor) * prev


@wp.func
def scale(x: wp.float32, lower: wp.float32, upper: wp.float32) -> wp.float32:
    return wp.float32(0.5) * (x + wp.float32(1.0)) * (upper - lower) + lower


@wp.func
def unscale(x: wp.float32, lower: wp.float32, upper: wp.float32) -> wp.float32:
    return (wp.float32(2.0) * x - upper - lower) / (upper - lower)


@wp.func
def randomize_rotation(rand0: wp.float32, rand1: wp.float32, x_axis: wp.vec3f, y_axis: wp.vec3f) -> wp.quatf:
    return wp.quat_from_axis_angle(x_axis, rand0 * wp.pi) * wp.quat_from_axis_angle(y_axis, rand1 * wp.pi)


@wp.func
def rotation_distance(object_rot: wp.quatf, target_rot: wp.quatf) -> wp.float32:
    """Angle [rad] between two ``(x, y, z, w)`` orientations, as :func:`~isaaclab.utils.math.quat_error_magnitude`."""
    quat_diff = object_rot * wp.quat_inverse(target_rot)
    v_norm = wp.length(wp.vec3f(quat_diff[0], quat_diff[1], quat_diff[2]))
    return wp.float32(2.0) * wp.asin(wp.min(v_norm, wp.float32(1.0)))


class ReorientDirectWarpEnv(DirectRLEnvWarp):
    """Warp twin of :class:`~isaaclab_tasks.core.reorient.reorient_direct_env.ReorientDirectEnv`."""

    cfg: AllegroHandEnvCfg

    def __init__(self, cfg: AllegroHandEnvCfg, render_mode: str | None = None, **kwargs):
        if cfg.asymmetric_obs:
            raise NotImplementedError("The Warp reorientation environment has no asymmetric critic observations.")
        super().__init__(cfg, render_mode, **kwargs)
        self.hand, self.object, self.goal_markers = [self.scene[name] for name in ("robot", "object", "goal_object")]

        # ---------------------------------------------------------------------
        # Constants
        # ---------------------------------------------------------------------

        # dof used for joint related init and sample
        self.num_hand_dofs = self.hand.num_joints

        # list of actuated joints
        actuated_dof_indices: list[int] = list()
        for joint_name in cfg.actuated_joint_names:
            actuated_dof_indices.append(self.hand.joint_names.index(joint_name))
        actuated_dof_indices.sort()
        self.num_actuated_dofs = len(actuated_dof_indices)
        self.actuated_dof_indices = wp.array(actuated_dof_indices, dtype=wp.int32, device=self.device)

        # finger bodies
        finger_bodies: list[int] = list()
        for body_name in self.cfg.fingertip_body_names:
            finger_bodies.append(self.hand.body_names.index(body_name))
        finger_bodies.sort()
        self.num_fingertips = len(finger_bodies)
        self.finger_bodies = wp.array(finger_bodies, dtype=wp.int32, device=self.device)

        # joint limits
        self.hand_dof_lower_limits = self.hand.data.joint_pos_limits_lower.warp
        self.hand_dof_upper_limits = self.hand.data.joint_pos_limits_upper.warp

        # unit vectors
        self.x_unit_vec = wp.vec3f(1.0, 0.0, 0.0)
        self.y_unit_vec = wp.vec3f(0.0, 1.0, 0.0)

        # Per-env origins (Warp view for kernels; Torch env uses `self.scene.env_origins` directly).
        self.env_origins = wp.from_torch(self.scene.env_origins, dtype=wp.vec3f)

        # ---------------------------------------------------------------------
        # Warp buffers
        # ---------------------------------------------------------------------

        # buffers for position targets
        self.prev_targets = wp.zeros((self.num_envs, self.num_hand_dofs), dtype=wp.float32, device=self.device)
        self.cur_targets = wp.zeros((self.num_envs, self.num_hand_dofs), dtype=wp.float32, device=self.device)

        # reset states sampled per env; the asset writers apply the masked rows
        self.reset_joint_pos = wp.zeros((self.num_envs, self.num_hand_dofs), dtype=wp.float32, device=self.device)
        self.reset_joint_vel = wp.zeros((self.num_envs, self.num_hand_dofs), dtype=wp.float32, device=self.device)
        self.reset_object_pose = wp.zeros(self.num_envs, dtype=wp.transformf, device=self.device)
        self.reset_object_vel = wp.zeros(self.num_envs, dtype=wp.spatial_vectorf, device=self.device)

        # per-step goal evaluation, written in `_get_dones` and reused by the reward
        self.orientation_error = wp.zeros(self.num_envs, dtype=wp.float32, device=self.device)
        self.goal_reached = wp.zeros(self.num_envs, dtype=wp.bool, device=self.device)

        # track goal resets
        self.reset_goal_buf = wp.zeros(self.num_envs, dtype=wp.bool, device=self.device)
        # used to compare object position
        self.in_hand_pos = wp.zeros(self.num_envs, dtype=wp.vec3f, device=self.device)
        # default goal positions
        self.goal_rot = wp.zeros(self.num_envs, dtype=wp.quatf, device=self.device)
        self.goal_pos = wp.zeros(self.num_envs, dtype=wp.vec3f, device=self.device)
        self.goal_pos_w = wp.zeros(self.num_envs, dtype=wp.vec3f, device=self.device)

        # Initialize goal constants from Torch (avoid a one-off kernel launch).
        in_hand_pos = self.object.data.default_root_pose.torch[:, 0:3] + torch.tensor(
            self.cfg.in_hand_pos_offset, device=self.device
        )
        self.in_hand_pos.assign(wp.from_torch(in_hand_pos, dtype=wp.vec3f))

        goal_pos = torch.tensor(self.cfg.goal_marker_position, device=self.device).repeat((self.num_envs, 1))
        self.goal_pos.assign(wp.from_torch(goal_pos, dtype=wp.vec3f))

        goal_rot = torch.zeros((self.num_envs, 4), device=self.device, dtype=torch.float32)
        goal_rot[:, 3] = 1.0  # (x, y, z, w)
        self.goal_rot.assign(wp.from_torch(goal_rot, dtype=wp.quatf))

        # Reduction buffers for consecutive_successes update (Warp-only).
        self._num_resets = wp.zeros(1, dtype=wp.float32, device=self.device)
        self._finished_cons_successes = wp.zeros(1, dtype=wp.float32, device=self.device)
        # track successes
        self.successes = wp.zeros(self.num_envs, dtype=wp.float32, device=self.device)
        self.consecutive_successes = wp.zeros(1, dtype=wp.float32, device=self.device)
        self._success_rate_stats = wp.zeros(2, dtype=wp.float32, device=self.device)
        self.success_rate = wp.zeros(1, dtype=wp.float32, device=self.device)

        # Persistent RL buffers (Warp).
        self.actions = wp.zeros((self.num_envs, self.cfg.action_space), dtype=wp.float32, device=self.device)
        self.observations = wp.zeros((self.num_envs, self.cfg.observation_space), dtype=wp.float32, device=self.device)
        self.rewards = wp.zeros((self.num_envs,), dtype=wp.float32, device=self.device)
        # Flag used as a print token for non-finite observations (Warp).
        self.obs_nonfinite_flag = wp.zeros(1, dtype=wp.int32, device=self.device)

        # Intermediate values (Warp) -- mirrors the Torch env's `_compute_intermediate_values` fields.
        self.fingertip_pos = wp.zeros((self.num_envs, self.num_fingertips), dtype=wp.vec3f, device=self.device)
        self.fingertip_rot = wp.zeros((self.num_envs, self.num_fingertips), dtype=wp.quatf, device=self.device)
        self.fingertip_velocities = wp.zeros(
            (self.num_envs, self.num_fingertips), dtype=wp.spatial_vectorf, device=self.device
        )

        self.object_pose = wp.zeros(self.num_envs, dtype=wp.transformf, device=self.device)
        self.object_vels = wp.zeros(self.num_envs, dtype=wp.spatial_vectorf, device=self.device)

        # ---------------------------------------------------------------------
        # Torch views / aliases
        # ---------------------------------------------------------------------

        # Bind torch buffers to warp buffers (same pattern as Warp Cartpole).
        self.torch_obs_buf = wp.to_torch(self.observations)
        self.torch_reward_buf = wp.to_torch(self.rewards)
        self.torch_reset_terminated = wp.to_torch(self.reset_terminated)
        self.torch_reset_time_outs = wp.to_torch(self.reset_time_outs)
        self.torch_episode_length_buf = self.episode_length_buf  # already a torch tensor via wp.to_torch

    def _pre_physics_step(self, actions: wp.array) -> None:
        # Store actions in a persistent Warp buffer (analogous to `actions.clone()` in the Torch env).
        wp.copy(self.actions, actions)

    @captured
    def _apply_action(self) -> None:
        wp.launch(
            apply_actions_to_targets,
            dim=(self.num_envs, self.num_actuated_dofs),
            inputs=[
                self.actions,
                self.hand_dof_lower_limits,
                self.hand_dof_upper_limits,
                self.actuated_dof_indices,
                self.cfg.act_moving_average,
                self.prev_targets,
                self.cur_targets,
            ],
            device=self.device,
        )

        # unactuated joints keep the targets their last reset wrote
        self.hand.actuators.target_command.set_position_mask(value=self.cur_targets)

    @captured
    def _get_observations(self) -> dict:
        if self.cfg.obs_type == "openai":
            self.compute_reduced_observations()
        elif self.cfg.obs_type == "full":
            self.compute_full_observations()
        else:
            raise ValueError(f"Unknown obs_type: {self.cfg.obs_type}")
        return {"policy": self.torch_obs_buf}

    @captured
    def _get_rewards(self) -> None:
        # Clear reduction buffers before launching the reward kernel.
        self._num_resets.zero_()
        self._finished_cons_successes.zero_()
        wp.launch(
            compute_rewards,
            dim=self.num_envs,
            inputs=[
                self.reset_buf,
                self.object_pose,
                self.in_hand_pos,
                self.goal_reached,
                self.orientation_error,
                self.cfg.dist_reward_scale,
                self.cfg.rot_reward_scale,
                self.cfg.rot_eps,
                self.actions,
                self.cfg.action_penalty_scale,
                self.cfg.reach_goal_bonus,
                self.cfg.fall_dist,
                self.cfg.fall_penalty,
                self.cfg.action_space,
                self.reset_goal_buf,
                self.successes,
                self._num_resets,
                self._finished_cons_successes,
                self.rewards,
            ],
            device=self.device,
        )

        # A separate kernel is needed as Warp does not support thread synchronization for reductions.
        wp.launch(
            update_consecutive_successes_from_stats,
            dim=1,
            inputs=[
                self._num_resets,
                self._finished_cons_successes,
                self.cfg.av_factor,
                self.consecutive_successes,
            ],
            device=self.device,
        )

        if "log" not in self.extras:
            self.extras["log"] = dict()
        # .mean() cannot be called here as it causes problems on stream
        self.extras["log"]["consecutive_successes"] = wp.to_torch(self.consecutive_successes)

        self._reset_target_pose(self.reset_goal_buf)

    @captured
    def _get_dones(self) -> None:
        self._compute_intermediate_values()

        wp.launch(
            get_dones,
            dim=self.num_envs,
            inputs=[
                self.max_episode_length,
                self.object_pose,
                self.in_hand_pos,
                self.goal_rot,
                self.cfg.fall_dist,
                self.cfg.success_tolerance,
                self.cfg.max_consecutive_success,
                self.successes,
                self._episode_length_buf_wp,
                self.orientation_error,
                self.goal_reached,
                self.reset_terminated,
                self.reset_time_outs,
                self.reset_buf,
            ],
            device=self.device,
        )

    @captured
    def _reset_idx(self, mask: wp.array | None = None):
        if mask is None:
            mask = self._ALL_ENV_MASK

        # resets articulation and rigid body attributes
        super()._reset_idx(mask)

        self._reset_target_pose(mask)

        wp.launch(
            reset_object,
            dim=self.num_envs,
            inputs=[
                self.object.data.default_root_pose.warp,
                self.env_origins,
                self.cfg.reset_position_noise,
                self.x_unit_vec,
                self.y_unit_vec,
                mask,
                WarpRng.state,
                self.reset_object_pose,
                self.reset_object_vel,
            ],
            device=self.device,
        )
        self.object.write_root_pose_to_sim_mask(root_pose=self.reset_object_pose, env_mask=mask)
        self.object.write_root_velocity_to_sim_mask(root_velocity=self.reset_object_vel, env_mask=mask)

        wp.launch(
            reset_hand,
            dim=self.num_envs,
            inputs=[
                self.hand.data.default_joint_pos.warp,
                self.hand.data.default_joint_vel.warp,
                self.hand_dof_lower_limits,
                self.hand_dof_upper_limits,
                self.cfg.reset_dof_pos_noise,
                self.cfg.reset_dof_vel_noise,
                mask,
                self.num_hand_dofs,
                WarpRng.state,
                self.reset_joint_pos,
                self.reset_joint_vel,
                self.prev_targets,
                self.cur_targets,
            ],
            device=self.device,
        )
        self.hand.actuators.target_command.set_position_mask(value=self.cur_targets, env_mask=mask)
        self.hand.write_joint_position_to_sim_mask(position=self.reset_joint_pos, env_mask=mask)
        self.hand.write_joint_velocity_to_sim_mask(velocity=self.reset_joint_vel, env_mask=mask)

        self._success_rate_stats.zero_()
        wp.launch(
            reset_successes,
            dim=self.num_envs,
            inputs=[mask, self.successes, self._success_rate_stats],
            device=self.device,
        )
        wp.launch(
            update_success_rate_from_stats,
            dim=1,
            inputs=[self._success_rate_stats, self.success_rate],
            device=self.device,
        )
        self.extras.setdefault("log", {})["Metrics/success_rate"] = wp.to_torch(self.success_rate)

        self._compute_intermediate_values()

    def _reset_target_pose(self, mask: wp.array):
        wp.launch(
            reset_target_pose,
            dim=self.num_envs,
            inputs=[
                mask,
                self.x_unit_vec,
                self.y_unit_vec,
                self.env_origins,
                self.goal_pos,
                WarpRng.state,
                self.goal_rot,
                self.reset_goal_buf,
                self.goal_pos_w,
            ],
            device=self.device,
        )

    def _post_step_visualize(self) -> None:
        """Update goal markers outside CUDA graph scope."""
        self.goal_markers.visualize(
            wp.to_torch(self.goal_pos_w),
            wp.to_torch(self.goal_rot),
            environment_ids=self.scene._ALL_INDICES,
        )

    def _compute_intermediate_values(self):
        # data for hand/object (Warp version of the Torch env's `_compute_intermediate_values`)
        wp.launch(
            compute_intermediate_values,
            dim=self.num_envs,
            inputs=[
                self.hand.data.body_pos_w.warp,
                self.hand.data.body_quat_w.warp,
                self.hand.data.body_vel_w.warp,
                self.finger_bodies,
                self.env_origins,
                self.object.data.root_link_pose_w.warp,
                self.object.data.root_com_vel_w.warp,
                self.num_fingertips,
                self.fingertip_pos,
                self.fingertip_rot,
                self.fingertip_velocities,
                self.object_pose,
                self.object_vels,
            ],
            device=self.device,
        )

    def compute_reduced_observations(self):
        # Per https://arxiv.org/pdf/1808.00177.pdf Table 2
        #   Fingertip positions
        #   Object Position, but not orientation
        #   Relative target orientation
        wp.launch(
            compute_reduced_observations,
            dim=self.num_envs,
            inputs=[
                self.fingertip_pos,
                self.object_pose,
                self.goal_rot,
                self.actions,
                self.num_fingertips,
                self.cfg.action_space,
                self.observations,
            ],
            device=self.device,
        )
        # Warp-native non-finite sanitization + print-once.
        wp.launch(
            sanitize_and_print_once,
            dim=(self.num_envs * self.cfg.observation_space),
            inputs=[self.observations.flatten(), self.obs_nonfinite_flag],
            device=self.device,
        )
        self.obs_nonfinite_flag.zero_()

    def compute_full_observations(self):
        wp.launch(
            compute_full_observations,
            dim=self.num_envs,
            inputs=[
                self.hand.data.joint_pos.warp,
                self.hand.data.joint_vel.warp,
                self.hand_dof_lower_limits,
                self.hand_dof_upper_limits,
                self.cfg.vel_obs_scale,
                self.object_pose,
                self.object_vels,
                self.in_hand_pos,
                self.goal_rot,
                self.fingertip_pos,
                self.fingertip_rot,
                self.fingertip_velocities,
                self.actions,
                self.num_hand_dofs,
                self.num_fingertips,
                self.cfg.action_space,
                self.observations,
            ],
            device=self.device,
        )
        # Warp-native non-finite sanitization + print-once.
        wp.launch(
            sanitize_and_print_once,
            dim=(self.num_envs * self.cfg.observation_space),
            inputs=[self.observations.flatten(), self.obs_nonfinite_flag],
            device=self.device,
        )
        self.obs_nonfinite_flag.zero_()
