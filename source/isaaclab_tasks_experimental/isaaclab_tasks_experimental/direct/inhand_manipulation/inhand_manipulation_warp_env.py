# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import warp as wp

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation  # , RigidObject
from isaaclab_experimental.envs import DirectRLEnvWarp
from isaaclab.markers import VisualizationMarkers
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

if TYPE_CHECKING:
    from isaaclab_tasks_experimental.direct.allegro_hand.allegro_hand_warp_env_cfg import AllegroHandWarpEnvCfg


@wp.kernel
def initialize_rng_state(
    # input
    seed: wp.int32,
    # output
    state: wp.array(dtype=wp.uint32),
):
    env_id = wp.tid()
    state[env_id] = wp.rand_init(seed, wp.int32(env_id))


@wp.kernel
def initialize_goal_constants(
    # input
    default_object_root_pose: wp.array(dtype=wp.transformf),
    # output
    in_hand_pos: wp.array(dtype=wp.vec3f),
    goal_pos: wp.array(dtype=wp.vec3f),
    goal_rot: wp.array(dtype=wp.quatf),
):
    env_id = wp.tid()
    p = wp.transform_get_translation(default_object_root_pose[env_id])
    in_hand_pos[env_id] = wp.vec3f(p[0], p[1], p[2] - wp.float32(0.04))
    goal_pos[env_id] = wp.vec3f(wp.float32(-0.2), wp.float32(-0.45), wp.float32(0.68))
    goal_rot[env_id] = wp.quatf(wp.float32(0.0), wp.float32(0.0), wp.float32(0.0), wp.float32(1.0))


@wp.kernel
def initialize_xyz_unit_vecs(
    # output
    x_unit_vecs: wp.array(dtype=wp.vec3f),
    y_unit_vecs: wp.array(dtype=wp.vec3f),
    z_unit_vecs: wp.array(dtype=wp.vec3f),
):
    env_id = wp.tid()
    x_unit_vecs[env_id] = wp.vec3f(wp.float32(1.0), wp.float32(0.0), wp.float32(0.0))
    y_unit_vecs[env_id] = wp.vec3f(wp.float32(0.0), wp.float32(1.0), wp.float32(0.0))
    z_unit_vecs[env_id] = wp.vec3f(wp.float32(0.0), wp.float32(0.0), wp.float32(1.0))


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
    x_unit_vecs: wp.array(dtype=wp.vec3f),
    y_unit_vecs: wp.array(dtype=wp.vec3f),
    env_origins: wp.array(dtype=wp.vec3f),
    goal_pos: wp.array(dtype=wp.vec3f),
    # input/output
    rng_state: wp.array(dtype=wp.uint32),
    # output
    goal_rot: wp.array(dtype=wp.quatf),
    reset_goal_buf: wp.array(dtype=wp.bool),
    goal_pos_w: wp.array(dtype=wp.vec3f),
):
    env_id = wp.tid()
    if env_mask[env_id]:
        rand0 = wp.randf(rng_state[env_id], wp.float32(-1.0), wp.float32(1.0))
        rng_state[env_id] += wp.uint32(1)
        rand1 = wp.randf(rng_state[env_id], wp.float32(-1.0), wp.float32(1.0))
        rng_state[env_id] += wp.uint32(1)

        goal_rot[env_id] = randomize_rotation(rand0, rand1, x_unit_vecs[env_id], y_unit_vecs[env_id])
        reset_goal_buf[env_id] = False

    # Warp-native addition: goal position in world frame.
    goal_pos_w[env_id] = goal_pos[env_id] + env_origins[env_id]


@wp.kernel
def reset_object(
    # input
    default_root_pose: wp.array(dtype=wp.transformf),
    env_origins: wp.array(dtype=wp.vec3f),
    reset_position_noise: wp.float32,
    x_unit_vecs: wp.array(dtype=wp.vec3f),
    y_unit_vecs: wp.array(dtype=wp.vec3f),
    env_mask: wp.array(dtype=wp.bool),
    # input/output
    rng_state: wp.array(dtype=wp.uint32),
    # output
    root_pose_w: wp.array(dtype=wp.transformf),
    root_vel_w: wp.array(dtype=wp.spatial_vectorf),
):
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
        rot_w = randomize_rotation(rand0, rand1, x_unit_vecs[env_id], y_unit_vecs[env_id])

        # The following should be equivalent, but consider using write_root_pose_to_sim and write_root_velocity_to_sim
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
    hand_dof_targets: wp.array2d(dtype=wp.float32),
):
    env_id = wp.tid()
    if env_mask[env_id]:
        # Each env runs sequentially inside this kernel (avoids RNG races across DOFs).
        for dof_id in range(num_dofs):
            dof_pos_noise = wp.randf(rng_state[env_id], wp.float32(0.0), wp.float32(1.0))
            rng_state[env_id] += wp.uint32(1)

            delta_max = upper_limits[env_id, dof_id] - default_joint_pos[env_id, dof_id]
            delta_min = lower_limits[env_id, dof_id] - default_joint_pos[env_id, dof_id]
            rand_delta = delta_min + (delta_max - delta_min) * dof_pos_noise
            pos = default_joint_pos[env_id, dof_id] + reset_dof_pos_noise * rand_delta


            dof_vel_noise = wp.randf(rng_state[env_id], wp.float32(-1.0), wp.float32(1.0))
            rng_state[env_id] += wp.uint32(1)
            vel = default_joint_vel[env_id, dof_id] + reset_dof_vel_noise * dof_vel_noise

            # The following lines should be equivalent to the following:
            # self.hand.write_joint_state_to_sim(dof_pos, dof_vel, env_ids=env_ids)
            joint_pos[env_id, dof_id] = pos
            joint_vel[env_id, dof_id] = vel

            prev_targets[env_id, dof_id] = pos
            cur_targets[env_id, dof_id] = pos
            hand_dof_targets[env_id, dof_id] = pos


@wp.kernel
def reset_successes(
    # input
    env_mask: wp.array(dtype=wp.bool),
    # output
    successes: wp.array(dtype=wp.float32),
):
    env_id = wp.tid()
    if env_mask[env_id]:
        successes[env_id] = wp.float32(0.0)


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
    object_pos: wp.array(dtype=wp.vec3f),
    object_rot: wp.array(dtype=wp.quatf),
    object_velocities: wp.array(dtype=wp.spatial_vectorf),
    object_linvel: wp.array(dtype=wp.vec3f),
    object_angvel: wp.array(dtype=wp.vec3f),
):
    env_id = wp.tid()

    for i in range(num_fingertips):
        body_id = finger_bodies[i]
        fingertip_pos[env_id, i] = body_pos_w[env_id, body_id] - env_origins[env_id]
        fingertip_rot[env_id, i] = body_quat_w[env_id, body_id]
        fingertip_velocities[env_id, i] = body_vel_w[env_id, body_id]

    p = wp.transform_get_translation(object_root_pose_w[env_id])
    object_pos[env_id] = p - env_origins[env_id]
    object_rot[env_id] = wp.transform_get_rotation(object_root_pose_w[env_id])
    object_velocities[env_id] = object_root_vel_w[env_id]
    object_linvel[env_id] = wp.vec3f(object_root_vel_w[env_id][0], object_root_vel_w[env_id][1], object_root_vel_w[env_id][2])
    object_angvel[env_id] = wp.vec3f(object_root_vel_w[env_id][3], object_root_vel_w[env_id][4], object_root_vel_w[env_id][5])


@wp.kernel
def get_dones(
    # input
    max_episode_length: wp.int32,
    object_pos: wp.array(dtype=wp.vec3f),
    object_rot: wp.array(dtype=wp.quatf),
    in_hand_pos: wp.array(dtype=wp.vec3f),
    goal_rot: wp.array(dtype=wp.quatf),
    fall_dist: wp.float32,
    success_tolerance: wp.float32,
    max_consecutive_success: wp.int32,
    successes: wp.array(dtype=wp.float32),
    # input/output
    episode_length_buf: wp.array(dtype=wp.int32),
    # output
    out_of_reach: wp.array(dtype=wp.bool),
    time_out: wp.array(dtype=wp.bool),
    reset: wp.array(dtype=wp.bool),
):
    env_id = wp.tid()

    goal_dist = wp.length(object_pos[env_id] - in_hand_pos[env_id])
    out_of_reach[env_id] = goal_dist >= fall_dist

    max_success_reached = False
    if max_consecutive_success > 0:
        # Reset progress (episode length buf) on goal envs if max_consecutive_success > 0
        rot_dist = rotation_distance(object_rot[env_id], goal_rot[env_id])
        if wp.abs(rot_dist) <= success_tolerance:
            episode_length_buf[env_id] = 0
        max_success_reached = successes[env_id] >= wp.float32(max_consecutive_success)

    time_out[env_id] = episode_length_buf[env_id] >= (max_episode_length - 1) or max_success_reached
    reset[env_id] = out_of_reach[env_id] or time_out[env_id]


@wp.kernel
def compute_reduced_observations(
    # input
    fingertip_pos: wp.array2d(dtype=wp.vec3f),
    object_pos: wp.array(dtype=wp.vec3f),
    object_rot: wp.array(dtype=wp.quatf),
    goal_rot: wp.array(dtype=wp.quatf),
    actions: wp.array2d(dtype=wp.float32),
    num_fingertips: wp.int32,
    action_dim: wp.int32,
    # output
    observations: wp.array2d(dtype=wp.float32),
):
    env_id = wp.tid()

    idx = int(0)
    for i in range(num_fingertips):
        observations[env_id, idx + 0] = fingertip_pos[env_id, i][0]
        observations[env_id, idx + 1] = fingertip_pos[env_id, i][1]
        observations[env_id, idx + 2] = fingertip_pos[env_id, i][2]
        idx += 3

    observations[env_id, idx + 0] = object_pos[env_id][0]
    observations[env_id, idx + 1] = object_pos[env_id][1]
    observations[env_id, idx + 2] = object_pos[env_id][2]
    idx += 3

    rel = quat_mul(object_rot[env_id], quat_conjugate(goal_rot[env_id]))
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
    object_pos: wp.array(dtype=wp.vec3f),
    object_rot: wp.array(dtype=wp.quatf),
    object_linvel: wp.array(dtype=wp.vec3f),
    object_angvel: wp.array(dtype=wp.vec3f),
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
    env_id = wp.tid()

    # hand
    for i in range(num_hand_dofs):
        observations[env_id, i] = unscale(hand_dof_pos[env_id, i], hand_dof_lower_limits[env_id, i], hand_dof_upper_limits[env_id, i])

    offset = num_hand_dofs
    for i in range(num_hand_dofs):
        observations[env_id, offset + i] = vel_obs_scale * hand_dof_vel[env_id, i]
    offset += num_hand_dofs

    # object
    observations[env_id, offset + 0] = object_pos[env_id][0]
    observations[env_id, offset + 1] = object_pos[env_id][1]
    observations[env_id, offset + 2] = object_pos[env_id][2]
    offset += 3

    observations[env_id, offset + 0] = object_rot[env_id][0]
    observations[env_id, offset + 1] = object_rot[env_id][1]
    observations[env_id, offset + 2] = object_rot[env_id][2]
    observations[env_id, offset + 3] = object_rot[env_id][3]
    offset += 4

    observations[env_id, offset + 0] = object_linvel[env_id][0]
    observations[env_id, offset + 1] = object_linvel[env_id][1]
    observations[env_id, offset + 2] = object_linvel[env_id][2]
    offset += 3

    observations[env_id, offset + 0] = vel_obs_scale * object_angvel[env_id][0]
    observations[env_id, offset + 1] = vel_obs_scale * object_angvel[env_id][1]
    observations[env_id, offset + 2] = vel_obs_scale * object_angvel[env_id][2]
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

    rel = quat_mul(object_rot[env_id], quat_conjugate(goal_rot[env_id]))
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
    object_pos: wp.array(dtype=wp.vec3f),
    object_rot: wp.array(dtype=wp.quatf),
    target_pos: wp.array(dtype=wp.vec3f),
    target_rot: wp.array(dtype=wp.quatf),
    dist_reward_scale: wp.float32,
    rot_reward_scale: wp.float32,
    rot_eps: wp.float32,
    actions: wp.array2d(dtype=wp.float32),
    action_penalty_scale: wp.float32,
    success_tolerance: wp.float32,
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
    env_id = wp.tid()

    goal_dist = wp.length(object_pos[env_id] - target_pos[env_id])
    rot_dist = rotation_distance(object_rot[env_id], target_rot[env_id])

    dist_rew = goal_dist * dist_reward_scale
    rot_rew = wp.float32(1.0) / (wp.abs(rot_dist) + rot_eps) * rot_reward_scale

    action_penalty = wp.float32(0.0)
    for i in range(action_dim):
        action_penalty += actions[env_id, i] * actions[env_id, i]

    # Total reward is: position distance + orientation alignment + action regularization + success bonus + fall penalty
    reward = dist_rew + rot_rew + action_penalty * action_penalty_scale

    # Find out which envs hit the goal and update successes count
    reached = wp.abs(rot_dist) <= success_tolerance
    goal_resets = reached or reset_goal_buf[env_id]
    reset_goal_buf[env_id] = goal_resets
    if goal_resets:
        successes[env_id] = successes[env_id] + wp.float32(1.0)

    # Success bonus: orientation is within `success_tolerance` of goal orientation
    if goal_resets:
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
def quat_conjugate(q: wp.quatf) -> wp.quatf:
    return wp.quatf(-q[0], -q[1], -q[2], q[3])


@wp.func
def quat_mul(q1: wp.quatf, q2: wp.quatf) -> wp.quatf:
    # Hamilton product for quaternions in (x, y, z, w).
    x1, y1, z1, w1 = q1[0], q1[1], q1[2], q1[3]
    x2, y2, z2, w2 = q2[0], q2[1], q2[2], q2[3]
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    return wp.quatf(x, y, z, w)


@wp.func
def quat_from_angle_axis(angle: wp.float32, axis: wp.vec3f) -> wp.quatf:
    # axis assumed to be unit-length in this task.
    half = angle * wp.float32(0.5)
    s = wp.sin(half)
    c = wp.cos(half)
    return wp.quatf(axis[0] * s, axis[1] * s, axis[2] * s, c)


@wp.func
def randomize_rotation(rand0: wp.float32, rand1: wp.float32, x_axis: wp.vec3f, y_axis: wp.vec3f) -> wp.quatf:
    return quat_mul(quat_from_angle_axis(rand0 * wp.pi, x_axis), quat_from_angle_axis(rand1 * wp.pi, y_axis))


@wp.func
def rotation_distance(object_rot: wp.quatf, target_rot: wp.quatf) -> wp.float32:
    # Orientation alignment for the cube in hand and goal cube
    quat_diff = quat_mul(object_rot, quat_conjugate(target_rot))
    # Match Torch env convention: uses indices [1:4] for the vector part (see `rotation_distance` in Torch env).
    v_norm = wp.sqrt(quat_diff[1] * quat_diff[1] + quat_diff[2] * quat_diff[2] + quat_diff[3] * quat_diff[3])
    v_norm = wp.min(v_norm, wp.float32(1.0))
    return wp.float32(2.0) * wp.asin(v_norm)


class InHandManipulationWarpEnv(DirectRLEnvWarp):
    cfg: AllegroHandWarpEnvCfg  # | ShadowHandWarpEnvCfg

    # def __init__(self, cfg: AllegroHandWarpEnvCfg | ShadowHandWarpEnvCfg, render_mode: str | None = None, **kwargs):
    def __init__(self, cfg: AllegroHandWarpEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

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

        # Warp index/mask helpers for kernels and articulation APIs.
        self.actuated_dof_indices = wp.array(actuated_dof_indices, dtype=wp.int32, device=self.device)
        actuated_mask = [False] * self.num_hand_dofs
        for idx in actuated_dof_indices:
            actuated_mask[idx] = True
        self.actuated_dof_mask = wp.array(actuated_mask, dtype=wp.bool, device=self.device)

        # finger bodies
        finger_bodies: list[int] = list()
        for body_name in self.cfg.fingertip_body_names:
            finger_bodies.append(self.hand.body_names.index(body_name))
        finger_bodies.sort()
        self.num_fingertips = len(finger_bodies)
        self.finger_bodies = wp.array(finger_bodies, dtype=wp.int32, device=self.device)

        # joint limits
        self.hand_dof_lower_limits = self.hand.data.joint_pos_limits_lower
        self.hand_dof_upper_limits = self.hand.data.joint_pos_limits_upper

        # unit vectors
        self.x_unit_vecs = wp.zeros(self.num_envs, dtype=wp.vec3f, device=self.device)
        self.y_unit_vecs = wp.zeros(self.num_envs, dtype=wp.vec3f, device=self.device)
        self.z_unit_vecs = wp.zeros(self.num_envs, dtype=wp.vec3f, device=self.device)
        wp.launch(
            initialize_xyz_unit_vecs,
            dim=self.num_envs,
            inputs=[
                self.x_unit_vecs,
                self.y_unit_vecs,
                self.z_unit_vecs,
            ],
            device=self.device,
        )

        # Per-env origins (Warp view for kernels; Torch env uses `self.scene.env_origins` directly).
        self.env_origins = wp.from_torch(self.scene.env_origins, dtype=wp.vec3f)

        # ---------------------------------------------------------------------
        # Warp buffers
        # ---------------------------------------------------------------------

        # buffers for position targets
        self.hand_dof_targets = wp.zeros((self.num_envs, self.num_hand_dofs), dtype=wp.float32, device=self.device)
        self.prev_targets = wp.zeros((self.num_envs, self.num_hand_dofs), dtype=wp.float32, device=self.device)
        self.cur_targets = wp.zeros((self.num_envs, self.num_hand_dofs), dtype=wp.float32, device=self.device)

        # track goal resets
        self.reset_goal_buf = wp.zeros(self.num_envs, dtype=wp.bool, device=self.device)
        # used to compare object position
        self.in_hand_pos = wp.zeros(self.num_envs, dtype=wp.vec3f, device=self.device)
        # default goal positions
        self.goal_rot = wp.zeros(self.num_envs, dtype=wp.quatf, device=self.device)
        self.goal_pos = wp.zeros(self.num_envs, dtype=wp.vec3f, device=self.device)
        self.goal_pos_w = wp.zeros(self.num_envs, dtype=wp.vec3f, device=self.device)
        # Initialize goal constants (in-hand reference pos, goal pose).
        wp.launch(
            initialize_goal_constants,
            dim=self.num_envs,
            inputs=[
                self.object.data.default_root_pose,
                self.in_hand_pos,
                self.goal_pos,
                self.goal_rot,
            ],
            device=self.device,
        )

        # initialize goal marker
        self.goal_markers = VisualizationMarkers(self.cfg.goal_object_cfg)

        # track successes
        self.successes = wp.zeros(self.num_envs, dtype=wp.float32, device=self.device)
        self.consecutive_successes = wp.zeros(1, dtype=wp.float32, device=self.device)
        # Reduction buffers for consecutive_successes update (Warp-only).
        self._num_resets = wp.zeros(1, dtype=wp.float32, device=self.device)
        self._finished_cons_successes = wp.zeros(1, dtype=wp.float32, device=self.device)

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

        self.object_pos = wp.zeros(self.num_envs, dtype=wp.vec3f, device=self.device)
        self.object_rot = wp.zeros(self.num_envs, dtype=wp.quatf, device=self.device)
        self.object_velocities = wp.zeros(self.num_envs, dtype=wp.spatial_vectorf, device=self.device)
        self.object_linvel = wp.zeros(self.num_envs, dtype=wp.vec3f, device=self.device)
        self.object_angvel = wp.zeros(self.num_envs, dtype=wp.vec3f, device=self.device)

        # RNG state (per-env) for randomizations in reset/goal resets.
        self.rng_state = wp.zeros(self.num_envs, dtype=wp.uint32, device=self.device)
        if self.cfg.seed is None:
            self.cfg.seed = -1
        wp.launch(
            initialize_rng_state,
            dim=self.num_envs,
            inputs=[
                self.cfg.seed,
                self.rng_state,
            ],
            device=self.device,
        )

        # ---------------------------------------------------------------------
        # Torch views / aliases
        # ---------------------------------------------------------------------

        # Bind torch buffers to warp buffers (same pattern as Warp Cartpole).
        self.torch_obs_buf = wp.to_torch(self.observations)
        self.torch_reward_buf = wp.to_torch(self.rewards)
        self.torch_reset_terminated = wp.to_torch(self.reset_terminated)
        self.torch_reset_time_outs = wp.to_torch(self.reset_time_outs)
        self.torch_episode_length_buf = wp.to_torch(self.episode_length_buf)

    def _setup_scene(self):
        # add hand, in-hand object, and goal object
        self.hand = Articulation(self.cfg.robot_cfg)
        self.object = Articulation(self.cfg.object_cfg)
        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # clone and replicate (no need to filter for this environment)
        self.scene.clone_environments(copy_from_source=False)
        # add articulation to scene - we must register to scene to randomize with EventManager
        self.scene.articulations["robot"] = self.hand
        self.scene.articulations["object"] = self.object
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: wp.array) -> None:
        # Store actions in a persistent Warp buffer (analogous to `actions.clone()` in the Torch env).
        wp.copy(self.actions, actions)

    def _apply_action(self) -> None:
        wp.launch(
            apply_actions_to_targets,
            dim=(self.num_envs, self.num_actuated_dofs),
            inputs=[
                self.actions,
                self.hand_dof_lower_limits,
                self.hand_dof_upper_limits,
                self.actuated_dof_indices,
                float(self.cfg.act_moving_average),
                self.prev_targets,
                self.cur_targets,
            ],
            device=self.device,
        )

        # Apply position targets, only set actuated joints
        self.hand.set_joint_position_target(self.cur_targets, joint_mask=self.actuated_dof_mask)

    def _get_observations(self) -> None:
        # if self.cfg.asymmetric_obs:
        #    self.fingertip_force_sensors = self.hand.root_physx_view.get_link_incoming_joint_force()[
        #        :, self.finger_bodies
        #    ]
        if self.cfg.obs_type == "openai":
            self.compute_reduced_observations()
        elif self.cfg.obs_type == "full":
            self.compute_full_observations()
        else:
            print("Unknown observations type!")

    def _get_rewards(self) -> None:
        # Clear reduction buffers before launching the reward kernel.
        # wp.assign(self._num_resets, 0.0)
        # wp.assign(self._finished_cons_successes, 0.0)
        self._num_resets.zero_()
        self._finished_cons_successes.zero_()
        wp.launch(
            compute_rewards,
            dim=self.num_envs,
            inputs=[
                self.reset_buf,
                self.object_pos,
                self.object_rot,
                self.in_hand_pos,
                self.goal_rot,
                float(self.cfg.dist_reward_scale),
                float(self.cfg.rot_reward_scale),
                float(self.cfg.rot_eps),
                self.actions,
                float(self.cfg.action_penalty_scale),
                float(self.cfg.success_tolerance),
                float(self.cfg.reach_goal_bonus),
                float(self.cfg.fall_dist),
                float(self.cfg.fall_penalty),
                int(self.cfg.action_space),
                self.reset_goal_buf,
                self.successes,
                self._num_resets,
                self._finished_cons_successes,
                self.rewards,
            ],
            device=self.device,
        )
        # Warp-native consecutive_successes reduction + EMA update (graphable).
        wp.launch(
            update_consecutive_successes_from_stats,
            dim=1,
            inputs=[
                self._num_resets,
                self._finished_cons_successes,
                float(self.cfg.av_factor),
                self.consecutive_successes,
            ],
            device=self.device,
        )

        # Reset goals for envs that reached the target (mask is `reset_goal_buf`).
        # This avoids Torch-side index extraction and keeps the step graphable.
        self._reset_target_pose(mask=self.reset_goal_buf)

    def _get_dones(self) -> None:
        self._compute_intermediate_values()

        wp.launch(
            get_dones,
            dim=self.num_envs,
            inputs=[
                self.max_episode_length,
                self.object_pos,
                self.object_rot,
                self.in_hand_pos,
                self.goal_rot,
                float(self.cfg.fall_dist),
                float(self.cfg.success_tolerance),
                int(self.cfg.max_consecutive_success),
                self.successes,
                self.episode_length_buf,
                self.reset_terminated,
                self.reset_time_outs,
                self.reset_buf,
            ],
            device=self.device,
        )

    def _reset_idx(self, mask: wp.array | None = None):
        if mask is None:
            mask = self._ALL_ENV_MASK

        # resets articulation and rigid body attributes
        super()._reset_idx(mask)

        # reset goals
        self._reset_target_pose(mask=mask)

        # reset object
        wp.launch(
            reset_object,
            dim=self.num_envs,
            inputs=[
                self.object.data.default_root_pose,
                self.env_origins,
                float(self.cfg.reset_position_noise),
                self.x_unit_vecs,
                self.y_unit_vecs,
                mask,
                self.rng_state,
                self.object.data.root_link_pose_w,
                self.object.data.root_com_vel_w,
            ],
            device=self.device,
        )

        # reset hand
        wp.launch(
            reset_hand,
            dim=self.num_envs,
            inputs=[
                self.hand.data.default_joint_pos,
                self.hand.data.default_joint_vel,
                self.hand_dof_lower_limits,
                self.hand_dof_upper_limits,
                float(self.cfg.reset_dof_pos_noise),
                float(self.cfg.reset_dof_vel_noise),
                mask,
                self.num_hand_dofs,
                self.rng_state,
                self.hand.data.joint_pos,
                self.hand.data.joint_vel,
                self.prev_targets,
                self.cur_targets,
                self.hand_dof_targets,
            ],
            device=self.device,
        )

        self.hand.set_joint_position_target(self.cur_targets, env_mask=mask)

        wp.launch(
            reset_successes,
            dim=self.num_envs,
            inputs=[
                mask,
                self.successes,
            ],
            device=self.device,
        )

        self._compute_intermediate_values()

    def _reset_target_pose(self, env_ids: Sequence[int] | None = None, mask: wp.array | None = None):
        # reset goal rotation
        if mask is None:
            if env_ids is None:
                return
            env_mask_list = [False] * self.num_envs
            for env_id in env_ids:
                env_mask_list[int(env_id)] = True
            mask = wp.array(env_mask_list, dtype=wp.bool, device=self.device)

        # update goal pose and markers
        wp.launch(
            reset_target_pose,
            dim=self.num_envs,
            inputs=[
                mask,
                self.x_unit_vecs,
                self.y_unit_vecs,
                self.env_origins,
                self.goal_pos,
                self.rng_state,
                self.goal_rot,
                self.reset_goal_buf,
                self.goal_pos_w,
            ],
            device=self.device,
        )

        # update goal pose and markers
        goal_pos = wp.to_torch(self.goal_pos_w)
        self.goal_markers.visualize(goal_pos, wp.to_torch(self.goal_rot))

    def _compute_intermediate_values(self):
        # data for hand/object (Warp version of the Torch env's `_compute_intermediate_values`)
        wp.launch(
            compute_intermediate_values,
            dim=self.num_envs,
            inputs=[
                self.hand.data.body_pos_w,
                self.hand.data.body_quat_w,
                self.hand.data.body_vel_w,
                self.finger_bodies,
                self.env_origins,
                self.object.data.root_link_pose_w,
                self.object.data.root_com_vel_w,
                self.num_fingertips,
                self.fingertip_pos,
                self.fingertip_rot,
                self.fingertip_velocities,
                self.object_pos,
                self.object_rot,
                self.object_velocities,
                self.object_linvel,
                self.object_angvel,
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
                self.object_pos,
                self.object_rot,
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
                self.hand.data.joint_pos,
                self.hand.data.joint_vel,
                self.hand_dof_lower_limits,
                self.hand_dof_upper_limits,
                float(self.cfg.vel_obs_scale),
                self.object_pos,
                self.object_rot,
                self.object_linvel,
                self.object_angvel,
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
