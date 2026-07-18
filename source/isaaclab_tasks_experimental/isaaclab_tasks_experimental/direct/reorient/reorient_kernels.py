# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Warp kernels and device math for the reorientation task family.

This is the family's math root: the experimental Direct environments and the
handover family (which reuses the shared hand idioms) launch these kernels on
the environment's Warp-native state (ProxyArray ``.warp`` views) and their own
pre-allocated output buffers. The module deliberately imports no environment
or mdp code.

A Direct environment step launches these kernels in the following order:

1. ``_pre_physics_step`` / ``_apply_action``: :func:`ema_actuation_kernel`
   turns the policy action into clamped joint position targets.
2. ``_get_dones``: :func:`out_of_reach_kernel` flags fallen objects, then
   :func:`reorient_success_kernel` evaluates goal orientation success once per
   step, and :func:`reorient_progress_kernel` folds it into the episode
   bookkeeping (episode-counter reset, time-out flags, minimum-error tracking).
3. ``_get_rewards``: :func:`reorient_reward` reuses the success state
   and launches :func:`reorient_reward_kernel` plus the two
   ``consecutive_success_*`` reduction kernels, all writing in place.
4. Observations: :func:`full_obs_kernel` or :func:`reduced_obs_kernel` build
   the flattened policy/critic vectors; the camera tasks derive cube corner
   keypoints with :func:`compute_cube_keypoints` /
   :func:`cube_keypoints_from_quat_kernel`.

Kernel signatures group their parameters by role, marked with ``# input``
(read only), ``# input/output`` (read and updated in place), and ``# output``
(written) comments.
"""

from __future__ import annotations

import warp as wp


@wp.kernel
def ema_actuation_kernel(
    # input
    actions: wp.array2d(dtype=wp.float32),
    lower: wp.array2d(dtype=wp.float32),
    upper: wp.array2d(dtype=wp.float32),
    dof_ids: wp.array(dtype=wp.int32),
    moving_average: float,
    # input/output
    prev_targets: wp.array2d(dtype=wp.float32),
    # output
    cur_targets: wp.array2d(dtype=wp.float32),
    compact_targets: wp.array2d(dtype=wp.float32),
):
    """Blend actions into clamped joint position targets with an exponential moving average.

    Launched over ``(num_envs, num_actuated_joints)``. Each normalized action in
    ``[-1, 1]`` is unscaled to its joint's position range [m or rad, depending on
    joint type], blended with the previous target (``moving_average`` weighs the
    new value), and clamped back to the limits. The result lands three ways:
    ``cur_targets``/``prev_targets`` at the full-joint index ``dof_ids[j]`` (so
    the smoothing state persists across steps), and ``compact_targets`` at the
    actuated-joint index ``j`` for the articulation's indexed target write.
    """
    i, j = wp.tid()
    dj = dof_ids[j]
    lo = lower[i, dj]
    hi = upper[i, dj]
    # unscale from [-1, 1] to the joint range, blend, and clamp
    t = 0.5 * (actions[i, j] + 1.0) * (hi - lo) + lo
    t = moving_average * t + (1.0 - moving_average) * prev_targets[i, dj]
    t = wp.clamp(t, lo, hi)
    cur_targets[i, dj] = t
    prev_targets[i, dj] = t
    compact_targets[i, j] = t


@wp.kernel
def fingertip_pos_kernel(
    # input
    body_pos_w: wp.array2d(dtype=wp.vec3f),
    env_origins: wp.array(dtype=wp.vec3f),
    body_ids: wp.array(dtype=wp.int32),
    # output
    out: wp.array2d(dtype=wp.float32),
):
    """Gather flattened fingertip positions [m] in the environment frame.

    Launched over ``(num_envs, num_fingertips)``; fingertip ``j`` fills columns
    ``[3j, 3j + 3)`` of ``out``.
    """
    i, j = wp.tid()
    p = body_pos_w[i, body_ids[j]] - env_origins[i]
    out[i, 3 * j + 0] = p[0]
    out[i, 3 * j + 1] = p[1]
    out[i, 3 * j + 2] = p[2]


@wp.kernel
def fingertip_quat_kernel(
    # input
    body_quat_w: wp.array2d(dtype=wp.quatf),
    body_ids: wp.array(dtype=wp.int32),
    # output
    out: wp.array2d(dtype=wp.float32),
):
    """Gather flattened fingertip ``(x, y, z, w)`` orientations.

    Launched over ``(num_envs, num_fingertips)``; fingertip ``j`` fills columns
    ``[4j, 4j + 4)`` of ``out``.
    """
    i, j = wp.tid()
    q = body_quat_w[i, body_ids[j]]
    out[i, 4 * j + 0] = q[0]
    out[i, 4 * j + 1] = q[1]
    out[i, 4 * j + 2] = q[2]
    out[i, 4 * j + 3] = q[3]


@wp.kernel
def fingertip_vel_kernel(
    # input
    body_vel_w: wp.array2d(dtype=wp.spatial_vectorf),
    body_ids: wp.array(dtype=wp.int32),
    # output
    out: wp.array2d(dtype=wp.float32),
):
    """Gather flattened fingertip spatial velocities [m/s, rad/s].

    Launched over ``(num_envs, num_fingertips)``; fingertip ``j`` fills columns
    ``[6j, 6j + 6)`` of ``out``.
    """
    i, j = wp.tid()
    v = body_vel_w[i, body_ids[j]]
    for k in range(6):
        out[i, 6 * j + k] = v[k]


# Column-parallel segment readers shared by the observation kernels launched over
# ``(num_envs, obs_dim)``: each maps a flattened segment column ``k`` to one value.


@wp.func
def fingertip_pos_col(
    body_pos_w: wp.array2d(dtype=wp.vec3f),
    env_origins: wp.array(dtype=wp.vec3f),
    body_ids: wp.array(dtype=wp.int32),
    i: int,
    k: int,
) -> float:
    """Column ``k`` of the flattened fingertip positions [m] in the environment frame."""
    p = body_pos_w[i, body_ids[k // 3]] - env_origins[i]
    return p[k % 3]


@wp.func
def fingertip_quat_col(
    body_quat_w: wp.array2d(dtype=wp.quatf),
    body_ids: wp.array(dtype=wp.int32),
    i: int,
    k: int,
) -> float:
    """Column ``k`` of the flattened fingertip ``(x, y, z, w)`` orientations."""
    return body_quat_w[i, body_ids[k // 4]][k % 4]


@wp.func
def fingertip_vel_col(
    body_vel_w: wp.array2d(dtype=wp.spatial_vectorf),
    body_ids: wp.array(dtype=wp.int32),
    i: int,
    k: int,
) -> float:
    """Column ``k`` of the flattened fingertip spatial velocities [m/s, rad/s]."""
    return body_vel_w[i, body_ids[k // 6]][k % 6]


# ---------------------------------------------------------------------------
# Success and reward math (Warp).
#
# Kernels shared by the Direct environments and the manager terms in
# :mod:`~isaaclab_tasks.core.reorient.mdp`; state arrays update in place. The
# quaternion arithmetic mirrors :func:`isaaclab.utils.math.quat_mul`
# (``(x, y, z, w)`` layout) term for term so results match the previously
# validated reference implementation.
# ---------------------------------------------------------------------------


@wp.func
def _rotation_distance(object_quat: wp.quatf, target_quat: wp.quatf) -> float:
    """Orientation distance [rad] between two ``(x, y, z, w)`` quaternions."""
    q1 = object_quat
    # quat_inverse == conjugate for these unit quaternions
    q2 = wp.quat_inverse(target_quat)
    ww = (q1[2] + q1[0]) * (q2[0] + q2[1])
    yy = (q1[3] - q1[1]) * (q2[3] + q2[2])
    zz = (q1[3] + q1[1]) * (q2[3] - q2[2])
    xx = ww + yy + zz
    qq = 0.5 * (xx + (q1[2] - q1[0]) * (q2[0] - q2[1]))
    x = qq - xx + (q1[0] + q1[3]) * (q2[0] + q2[3])
    y = qq - yy + (q1[3] - q1[0]) * (q2[1] + q2[2])
    z = qq - zz + (q1[2] + q1[1]) * (q2[3] - q2[0])
    return 2.0 * wp.asin(wp.min(wp.length(wp.vec3(x, y, z)), 1.0))


@wp.kernel
def rotation_distance_kernel(
    # input
    object_quat: wp.array(dtype=wp.quatf),
    target_quat: wp.array(dtype=wp.quatf),
    # output
    distance: wp.array(dtype=wp.float32),
):
    """Per-environment orientation distance [rad] between object and target."""
    i = wp.tid()
    distance[i] = _rotation_distance(object_quat[i], target_quat[i])


@wp.kernel
def reorient_success_kernel(
    # input
    object_quat: wp.array(dtype=wp.quatf),
    target_quat: wp.array(dtype=wp.quatf),
    success_tolerance: float,
    # output
    success: wp.array(dtype=wp.bool),
    distance: wp.array(dtype=wp.float32),
):
    """Evaluate goal-orientation success while exposing the physical error.

    The single per-step success evaluation: writes the orientation distance
    [rad] and whether it is within ``success_tolerance`` [rad]. Downstream
    consumers — :func:`reorient_progress_kernel`, the manager command/reward/
    termination terms, and the environments' metrics — reuse these buffers
    instead of recomputing the quaternion math.
    """
    i = wp.tid()
    d = _rotation_distance(object_quat[i], target_quat[i])
    distance[i] = d
    success[i] = d <= success_tolerance


@wp.kernel
def reorient_reward_kernel(
    # input
    object_pos_w: wp.array(dtype=wp.vec3f),
    env_origins: wp.array(dtype=wp.vec3f),
    object_quat: wp.array(dtype=wp.quatf),
    target_pos_e: wp.array(dtype=wp.vec3f),
    target_quat: wp.array(dtype=wp.quatf),
    actions: wp.array2d(dtype=wp.float32),
    reset_buf: wp.array(dtype=wp.bool),
    distance_scale: float,
    rotation_scale: float,
    rotation_epsilon: float,
    action_penalty_scale: float,
    success_tolerance: float,
    success_bonus: float,
    fall_distance: float,
    fall_penalty: float,
    # input/output
    goal_resets: wp.array(dtype=wp.bool),
    successes: wp.array(dtype=wp.float32),
    # output
    reward: wp.array(dtype=wp.float32),
    resets: wp.array(dtype=wp.bool),
):
    """Per-environment reorientation reward and success state transition.

    The reward sums four terms: goal distance [m] times ``distance_scale``
    (negative scale penalizes distance), the inverse orientation error
    ``rotation_scale / (distance [rad] + rotation_epsilon)``, the squared-action
    penalty times ``action_penalty_scale``, plus a one-time ``success_bonus``
    when the goal orientation is first within ``success_tolerance`` [rad] and a
    ``fall_penalty`` when the object drifts beyond ``fall_distance`` [m].

    State transition, in place: ``goal_resets`` marks environments due for goal
    resampling (sticky until the caller resamples and clears it), each newly
    reached goal increments ``successes``, and ``resets`` combines the incoming
    episode resets with the fallen-object condition for the consecutive-success
    reduction downstream.
    """
    i = wp.tid()
    # position error to the goal [m]
    # object position is world-frame; the target is authored in the environment frame
    goal_distance = wp.length(object_pos_w[i] - env_origins[i] - target_pos_e[i])
    rotation_distance = _rotation_distance(object_quat[i], target_quat[i])
    # a goal within tolerance marks this env for goal resampling (sticky until resampled)
    goal_reset = rotation_distance <= success_tolerance or goal_resets[i]
    action_penalty = float(0.0)
    for j in range(actions.shape[1]):
        action_penalty += actions[i, j] * actions[i, j]
    value = (
        goal_distance * distance_scale
        + rotation_scale / (rotation_distance + rotation_epsilon)
        + action_penalty * action_penalty_scale
    )
    # one-time bonus on reaching the goal; penalty and episode reset when the object falls
    if goal_reset:
        value += success_bonus
    fell = goal_distance >= fall_distance
    if fell:
        value += fall_penalty
    reward[i] = value
    goal_resets[i] = goal_reset
    # per-episode success count feeds the consecutive-success moving average
    if goal_reset:
        successes[i] = successes[i] + 1.0
    resets[i] = fell or reset_buf[i]


@wp.kernel
def consecutive_success_stats_kernel(
    # input
    successes: wp.array(dtype=wp.float32),
    resets: wp.array(dtype=wp.bool),
    # output
    stats: wp.array(dtype=wp.float32),
):
    """Accumulate the resetting-env count and their episode successes into ``stats``.

    First half of the consecutive-success moving-average reduction (atomic
    adds). ``stats`` must be zeroed before the launch; ``[0]`` receives the
    number of resetting environments and ``[1]`` the sum of their episode
    success counts.
    """
    i = wp.tid()
    if resets[i]:
        wp.atomic_add(stats, 0, 1.0)
        wp.atomic_add(stats, 1, successes[i])


@wp.kernel
def consecutive_success_update_kernel(
    # input
    stats: wp.array(dtype=wp.float32),
    averaging_factor: float,
    # input/output
    consecutive_successes: wp.array(dtype=wp.float32),
):
    """Fold the finished-episode statistics into the moving average.

    Second half of the reduction; launch with ``dim=1`` after
    :func:`consecutive_success_stats_kernel` so the completed sums are visible.
    Leaves the average untouched when no episode finished this step.
    """
    if stats[0] > 0.0:
        consecutive_successes[0] = (
            averaging_factor * stats[1] / stats[0] + (1.0 - averaging_factor) * consecutive_successes[0]
        )


class ReorientRewardBuffers:
    """Caller-owned device buffers for :func:`reorient_reward`.

    Allocate once (Direct environments in ``__init__``, manager terms in their
    term ``__init__``); every step writes in place.
    """

    def __init__(self, num_envs: int, device: str):
        device = str(device)
        self.reward = wp.zeros(num_envs, dtype=wp.float32, device=device)
        self.resets = wp.empty(num_envs, dtype=wp.bool, device=device)
        # scratch for the consecutive-success reduction: [0] resetting-env
        # count, [1] summed episode successes of the resetting environments
        self.stats = wp.zeros(2, dtype=wp.float32, device=device)


def reorient_reward(
    reset_buf: wp.array,
    goal_resets: wp.array,
    successes: wp.array,
    consecutive_successes: wp.array,
    object_pos_w: wp.array,
    env_origins: wp.array,
    object_quat: wp.array,
    target_pos_e: wp.array,
    target_quat: wp.array,
    actions: wp.array,
    distance_scale: float,
    rotation_scale: float,
    rotation_epsilon: float,
    action_penalty_scale: float,
    success_tolerance: float,
    success_bonus: float,
    fall_distance: float,
    fall_penalty: float,
    averaging_factor: float,
    buffers: ReorientRewardBuffers,
) -> wp.array:
    """Compute the Direct reorientation reward and success state transition in place.

    Updates :paramref:`goal_resets`, :paramref:`successes`, and
    :paramref:`consecutive_successes` in place and writes the per-environment
    reward into ``buffers.reward``.

    Args:
        reset_buf: Episode-reset flags, ``wp.bool``. The kernel ORs in the
            fallen-object condition itself, so callers whose only other reset
            source is the episode time-out may pass those flags directly.
        goal_resets: Goal-reset flags, ``wp.bool``; read and updated in place.
        successes: Goals reached in each episode, ``wp.float32``; updated in place.
        consecutive_successes: One-element moving-average success count,
            ``wp.float32``; updated in place.
        object_pos_w: Object world positions [m], ``wp.vec3f``.
        env_origins: Environment origins [m], ``wp.vec3f``.
        object_quat: Object ``(x, y, z, w)`` orientations, ``wp.quatf``.
        target_pos_e: Goal positions in the environment frame [m], ``wp.vec3f``.
        target_quat: Goal ``(x, y, z, w)`` orientations, ``wp.quatf``.
        actions: Normalized joint actions, 2D ``wp.float32``.
        distance_scale: Position-distance reward scale [1/m].
        rotation_scale: Orientation reward scale [rad].
        rotation_epsilon: Orientation reward regularizer [rad].
        action_penalty_scale: Squared-action reward scale.
        success_tolerance: Goal orientation tolerance [rad].
        success_bonus: Reward added when a goal is reached.
        fall_distance: Object-to-goal termination distance [m].
        fall_penalty: Reward added when the object is out of reach.
        averaging_factor: Consecutive-success moving-average factor.
        buffers: Caller-owned output/scratch buffers.

    Returns:
        The per-environment reward, a view of ``buffers.reward``.
    """
    num_envs = buffers.reward.shape[0]
    wp.launch(
        reorient_reward_kernel,
        dim=num_envs,
        inputs=[
            object_pos_w,
            env_origins,
            object_quat,
            target_pos_e,
            target_quat,
            actions,
            reset_buf,
            distance_scale,
            rotation_scale,
            rotation_epsilon,
            action_penalty_scale,
            success_tolerance,
            success_bonus,
            fall_distance,
            fall_penalty,
        ],
        outputs=[
            goal_resets,
            successes,
            buffers.reward,
            buffers.resets,
        ],
        device=buffers.reward.device,
    )
    buffers.stats.zero_()
    wp.launch(
        consecutive_success_stats_kernel,
        dim=num_envs,
        inputs=[successes, buffers.resets],
        outputs=[buffers.stats],
        device=buffers.reward.device,
    )
    wp.launch(
        consecutive_success_update_kernel,
        dim=1,
        inputs=[buffers.stats, averaging_factor],
        outputs=[consecutive_successes],
        device=buffers.reward.device,
    )
    return buffers.reward


# ---------------------------------------------------------------------------
# Cube-keypoint math shared by the Direct camera environment and the manager
# camera observation terms.
# ---------------------------------------------------------------------------


CUBE_HALF_SIZE = wp.vec3(0.03, 0.03, 0.03)
"""Half side lengths [m] of the reorientation cube."""


@wp.kernel
def cube_keypoints_from_quat_kernel(
    # input
    quat: wp.array(dtype=wp.quatf),
    half_size: wp.vec3,
    # output
    keypoints: wp.array2d(dtype=wp.float32),
):
    """Rotation-only cube-corner offsets [m] from batched orientations.

    Launched over ``(num_envs, num_corners)``: corner ``c`` (its index bits
    select the +/- half side per axis) fills the flattened columns
    ``[3c, 3c + 3)``. Used where only the rotated corner offsets matter, e.g.
    goal-orientation keypoints without a translation.
    """
    env, corner = wp.tid()
    # corner index bits select the +/- half-side per axis (bit set -> negative)
    sign_x = wp.where(((corner >> 0) & 1) == 0, 1.0, -1.0)
    sign_y = wp.where(((corner >> 1) & 1) == 0, 1.0, -1.0)
    sign_z = wp.where(((corner >> 2) & 1) == 0, 1.0, -1.0)
    offset = wp.vec3(sign_x * half_size[0], sign_y * half_size[1], sign_z * half_size[2])
    p = wp.quat_rotate(quat[env], offset)
    keypoints[env, 3 * corner + 0] = p[0]
    keypoints[env, 3 * corner + 1] = p[1]
    keypoints[env, 3 * corner + 2] = p[2]


@wp.kernel
def _cube_keypoints_kernel(
    # input
    pose: wp.array(dtype=wp.float32, ndim=2),
    half_size: wp.vec3,
    # output
    keypoints: wp.array2d(dtype=wp.vec3),
):
    """Posed cube-corner positions [m]; launch via :func:`compute_cube_keypoints`."""
    env, corner = wp.tid()
    # corner index bits select the +/- half-side per axis (bit set -> negative)
    sign_x = wp.where(((corner >> 0) & 1) == 0, 1.0, -1.0)
    sign_y = wp.where(((corner >> 1) & 1) == 0, 1.0, -1.0)
    sign_z = wp.where(((corner >> 2) & 1) == 0, 1.0, -1.0)
    offset = wp.vec3(sign_x * half_size[0], sign_y * half_size[1], sign_z * half_size[2])
    orientation = wp.quat(pose[env, 3], pose[env, 4], pose[env, 5], pose[env, 6])
    position = wp.vec3(pose[env, 0], pose[env, 1], pose[env, 2])
    keypoints[env, corner] = position + wp.quat_rotate(orientation, offset)


def compute_cube_keypoints(
    pose: wp.array,
    keypoints: wp.array,
    size: tuple[float, float, float] = (2 * 0.03, 2 * 0.03, 2 * 0.03),
) -> None:
    """Compute cube-corner positions for batched poses.

    Args:
        pose: Cube center poses ``(x, y, z, qx, qy, qz, qw)`` [m, unit
            quaternion], 2D ``wp.float32``.
        keypoints: Output corner positions [m], 2D ``wp.vec3`` of shape
            ``(num_envs, num_keypoints)``; the corner count is taken from it.
        size: Cube side lengths along each axis [m].
    """
    wp.launch(
        _cube_keypoints_kernel,
        dim=(keypoints.shape[0], keypoints.shape[1]),
        inputs=[
            pose,
            wp.vec3(size[0] / 2.0, size[1] / 2.0, size[2] / 2.0),
        ],
        outputs=[keypoints],
        device=keypoints.device,
    )


@wp.kernel
def out_of_reach_kernel(
    # input
    object_pos_w: wp.array(dtype=wp.vec3f),
    env_origins: wp.array(dtype=wp.vec3f),
    target_pos_e: wp.array(dtype=wp.vec3f),
    fall_distance: float,
    # output
    out_of_reach: wp.array(dtype=wp.bool),
):
    """Flag environments whose object drifted beyond ``fall_distance`` [m] from the in-hand target."""
    i = wp.tid()
    out_of_reach[i] = wp.length(object_pos_w[i] - env_origins[i] - target_pos_e[i]) >= fall_distance


@wp.kernel
def full_obs_kernel(
    # input
    joint_pos: wp.array2d(dtype=wp.float32),
    joint_vel: wp.array2d(dtype=wp.float32),
    lower: wp.array2d(dtype=wp.float32),
    upper: wp.array2d(dtype=wp.float32),
    object_pos_w: wp.array(dtype=wp.vec3f),
    env_origins: wp.array(dtype=wp.vec3f),
    object_quat: wp.array(dtype=wp.quatf),
    object_lin_vel: wp.array(dtype=wp.vec3f),
    object_ang_vel: wp.array(dtype=wp.vec3f),
    in_hand_pos_e: wp.array(dtype=wp.vec3f),
    goal_quat: wp.array(dtype=wp.quatf),
    body_pos_w: wp.array2d(dtype=wp.vec3f),
    body_quat_w: wp.array2d(dtype=wp.quatf),
    body_vel_w: wp.array2d(dtype=wp.spatial_vectorf),
    finger_ids: wp.array(dtype=wp.int32),
    force: wp.array2d(dtype=wp.vec3f),
    torque: wp.array2d(dtype=wp.vec3f),
    wrench_ids: wp.array(dtype=wp.int32),
    actions: wp.array2d(dtype=wp.float32),
    vel_scale: float,
    force_scale: float,
    with_forces: int,
    # output
    out: wp.array2d(dtype=wp.float32),
):
    """Direct full observation / full state, matching the reference concatenation order.

    Column layout of ``out``, for ``J`` joints, ``F`` fingertips, and ``A``
    actions (segment widths in parentheses):

    1. normalized joint positions (``J``), scaled joint velocities (``J``)
    2. object: environment-frame position (3), ``(x, y, z, w)`` rotation (4),
       linear velocity (3), scaled angular velocity (3)
    3. goal: in-hand anchor position (3), goal rotation (4), object-to-goal
       rotation difference (4)
    4. fingertips: positions (``3F``), rotations (``4F``), spatial
       velocities (``6F``)
    5. fingertip wrenches (``6F``) — full state only; the segment exists when
       ``with_forces != 0`` and zero-fills while the sensor has no data
       (``with_forces == -1``)
    6. last actions (``A``)

    Launched over ``(num_envs, obs_dim)``: each thread walks a branch ladder over the
    segment boundaries and writes one output column, so warps (32 consecutive columns)
    stay branch-uniform except at segment boundaries.
    """
    i, j = wp.tid()
    num_joints = joint_pos.shape[1]
    num_fingers = finger_ids.shape[0]
    # segment boundaries, in column order
    end_joint = 2 * num_joints
    end_object = end_joint + 13
    end_goal = end_object + 11
    end_tip_pos = end_goal + 3 * num_fingers
    end_tip_quat = end_tip_pos + 4 * num_fingers
    end_tip_vel = end_tip_quat + 6 * num_fingers
    end_wrench = end_tip_vel
    if with_forces != 0:
        end_wrench += 6 * num_fingers
    # hand: normalized DOF positions, scaled DOF velocities
    if j < num_joints:
        out[i, j] = 2.0 * (joint_pos[i, j] - lower[i, j]) / (upper[i, j] - lower[i, j]) - 1.0
    elif j < end_joint:
        out[i, j] = vel_scale * joint_vel[i, j - num_joints]
    # object pose and velocities (environment frame position)
    elif j < end_object:
        k = j - end_joint
        if k < 3:
            p = object_pos_w[i] - env_origins[i]
            out[i, j] = p[k]
        elif k < 7:
            out[i, j] = object_quat[i][k - 3]
        elif k < 10:
            out[i, j] = object_lin_vel[i][k - 7]
        else:
            out[i, j] = vel_scale * object_ang_vel[i][k - 10]
    # goal: in-hand anchor, goal rotation, and the goal-to-object rotation difference
    elif j < end_goal:
        k = j - end_object
        if k < 3:
            out[i, j] = in_hand_pos_e[i][k]
        elif k < 7:
            out[i, j] = goal_quat[i][k - 3]
        else:
            # quat_inverse == conjugate for these unit quaternions, matching
            # isaaclab.utils.math.quat_mul/quat_conjugate semantics
            qe = object_quat[i] * wp.quat_inverse(goal_quat[i])
            out[i, j] = qe[k - 7]
    # fingertips: environment-frame positions, rotations, spatial velocities
    elif j < end_tip_pos:
        out[i, j] = fingertip_pos_col(body_pos_w, env_origins, finger_ids, i, j - end_goal)
    elif j < end_tip_quat:
        out[i, j] = fingertip_quat_col(body_quat_w, finger_ids, i, j - end_tip_pos)
    elif j < end_tip_vel:
        out[i, j] = fingertip_vel_col(body_vel_w, finger_ids, i, j - end_tip_quat)
    # fingertip force/torque sensors (full state only; absent when with_forces == 0)
    elif j < end_wrench:
        if with_forces == 1:
            k = j - end_tip_vel
            c = k % 6
            if c < 3:
                out[i, j] = force_scale * force[i, wrench_ids[k // 6]][c]
            else:
                out[i, j] = force_scale * torque[i, wrench_ids[k // 6]][c - 3]
        else:
            # full state requested but the sensor has no data yet: zero block
            out[i, j] = 0.0
    # actions
    else:
        out[i, j] = actions[i, j - end_wrench]


@wp.kernel
def reduced_obs_kernel(
    # input
    body_pos_w: wp.array2d(dtype=wp.vec3f),
    env_origins: wp.array(dtype=wp.vec3f),
    finger_ids: wp.array(dtype=wp.int32),
    object_pos_w: wp.array(dtype=wp.vec3f),
    object_quat: wp.array(dtype=wp.quatf),
    goal_quat: wp.array(dtype=wp.quatf),
    actions: wp.array2d(dtype=wp.float32),
    # output
    out: wp.array2d(dtype=wp.float32),
):
    """Direct reduced (OpenAI) observation, matching the reference concatenation order.

    Column layout of ``out``, for ``F`` fingertips and ``A`` actions:
    environment-frame fingertip positions (``3F``), then the object's
    environment-frame position (3) and object-to-goal rotation difference (4),
    then the last actions (``A``). One thread per environment writes the whole
    row.
    """
    i = wp.tid()
    num_fingers = finger_ids.shape[0]
    for f in range(num_fingers):
        fp = body_pos_w[i, finger_ids[f]] - env_origins[i]
        out[i, 3 * f + 0] = fp[0]
        out[i, 3 * f + 1] = fp[1]
        out[i, 3 * f + 2] = fp[2]
    idx = 3 * num_fingers
    p = object_pos_w[i] - env_origins[i]
    # quat_inverse == conjugate for these unit quaternions, matching
    # isaaclab.utils.math.quat_mul/quat_conjugate semantics
    qe = object_quat[i] * wp.quat_inverse(goal_quat[i])
    out[i, idx + 0] = p[0]
    out[i, idx + 1] = p[1]
    out[i, idx + 2] = p[2]
    out[i, idx + 3] = qe[0]
    out[i, idx + 4] = qe[1]
    out[i, idx + 5] = qe[2]
    out[i, idx + 6] = qe[3]
    idx += 7
    for a in range(actions.shape[1]):
        out[i, idx + a] = actions[i, a]


@wp.kernel
def reorient_progress_kernel(
    # input
    success: wp.array(dtype=wp.bool),
    distance: wp.array(dtype=wp.float32),
    successes: wp.array(dtype=wp.float32),
    max_consecutive_success: float,
    max_episode_length: wp.int64,
    # input/output
    episode_length: wp.array(dtype=wp.int64),
    minimum_error: wp.array(dtype=wp.float32),
    # output
    time_out: wp.array(dtype=wp.bool),
    has_sample: wp.array(dtype=wp.bool),
):
    """Per-step progress bookkeeping downstream of :func:`reorient_success_kernel`.

    Consumes the success flags and orientation distances [rad] and, in place:
    zeroes the episode progress on goal-reached environments (only when
    :paramref:`max_consecutive_success` > 0), flags episode time-out — by
    elapsed episode length or by reaching the consecutive-success cap — and
    tightens the episode-minimum error tracking (absorbing
    ``EpisodeErrorRecorder.update``: non-finite distances leave
    ``minimum_error`` and the sticky ``has_sample`` marker untouched).
    """
    i = wp.tid()
    max_success_reached = False
    if max_consecutive_success > 0.0:
        if success[i]:
            episode_length[i] = wp.int64(0)
        max_success_reached = successes[i] >= max_consecutive_success
    time_out[i] = episode_length[i] >= max_episode_length - wp.int64(1) or max_success_reached
    d = distance[i]
    if wp.isfinite(d):
        if d < minimum_error[i]:
            minimum_error[i] = d
        has_sample[i] = True
