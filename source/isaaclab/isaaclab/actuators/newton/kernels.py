# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared Warp kernels for the Newton actuator fast path."""

from typing import Any

import torch
import warp as wp

from isaaclab.actuators import ActuatorBase, ImplicitActuator

# ---------------------------------------------------------------------------
# Adapter / per-actuator helper kernels: per-DOF zeroing, env-mask building,
# per-DOF env-mask projection (used by :meth:`NewtonActuatorAdapter.reset`),
# and a partial scatter that overwrites only the cells in a
# (env_ids × joint_ids) sub-grid of a Newton ``Actuator``'s component
# parameter array. Shared by every backend through
# ``write_newton_actuator_parameter`` in :mod:`.adapter`.
# ---------------------------------------------------------------------------


@wp.kernel(enable_backward=False)
def zero_at_indices_kernel(data: wp.array(dtype=wp.float32), indices: wp.array(dtype=wp.uint32)):
    """Zero a flat ``data`` buffer at the given flat ``indices``."""
    i = wp.tid()
    data[indices[i]] = 0.0


@wp.kernel(enable_backward=False)
def set_mask_kernel(mask: wp.array(dtype=wp.bool), indices: wp.array(dtype=wp.int32)):
    """Set ``mask[indices[i]] = True`` for each ``i``. The mask must be pre-zeroed."""
    i = wp.tid()
    mask[indices[i]] = True


@wp.kernel(enable_backward=False)
def build_per_dof_env_mask_kernel(
    indices: wp.array(dtype=wp.uint32),
    env_mask: wp.array(dtype=wp.bool),
    dof_offset: int,
    num_joints: int,
    out_mask: wp.array(dtype=wp.bool),
):
    """Build a per-DOF mask from a per-env mask, for one Newton actuator.

    Newton's :meth:`Actuator.State.reset` expects a mask of length
    ``num_actuators`` (= ``num_envs * dofs_per_actuator``). Each entry
    gates the corresponding column of the actuator's state buffers. This
    kernel maps a per-env boolean mask onto that per-DOF layout via the
    actuator's flat ``indices``.
    """
    i = wp.tid()
    global_dof = int(indices[i]) - dof_offset
    env = global_dof // num_joints
    out_mask[i] = env_mask[env]


@wp.kernel(enable_backward=False)
def scatter_gain_kernel(
    src: wp.array(dtype=Any),
    dst: wp.array(dtype=Any),
    indices: wp.array(dtype=wp.uint32),
    dof_offset: int,
    num_envs: int,
    num_joints: int,
    env_stride: int,
):
    """Scatter per-actuator ``src`` values into a flat per-env-per-DOF ``dst``.

    A Newton actuator can span multiple articulations. Indices outside the
    requested articulation range are ignored.

    Args:
        src: Per-actuator parameter values (e.g. ``controller.kp``).
        dst: Flat ``(num_envs * num_joints)`` articulation-local snapshot buffer.
        indices: Actuator's flat env-major global DOF indices.
        dof_offset: Offset of this articulation's DOFs within each environment.
        num_envs: Number of articulation environments.
        num_joints: Articulation-local joint count (``dst``'s inner stride).
        env_stride: Whole-model per-env DOF count (the stride used to build
            ``indices``).
    """
    i = wp.tid()
    global_dof = int(indices[i])
    env = global_dof // env_stride
    local_dof = global_dof - env * env_stride - dof_offset
    if env < num_envs and local_dof >= 0 and local_dof < num_joints:
        dst[env * num_joints + local_dof] = src[i]


@wp.kernel(enable_backward=False)
def patch_actuator_param_kernel(
    indices: wp.array(dtype=wp.uint32),
    env_id_pos: wp.array(dtype=wp.int32),
    joint_id_pos: wp.array(dtype=wp.int32),
    values: wp.array2d(dtype=Any),
    dof_offset: int,
    num_envs: int,
    num_joints: int,
    env_stride: int,
    dst: wp.array(dtype=Any),
):
    """Per-actuator scatter for partial parameter updates.

    For each slot ``i`` in the actuator's flat env-major ``indices``, derive
    the (env, backend-local joint) pair and, when ``env_id_pos[env]`` and
    ``joint_id_pos[joint]`` select a cell of ``values`` (entries are ``-1``
    outside the ``(len(env_ids), len(joint_ids))`` selection), overwrite
    ``dst[i]`` — the per-actuator component parameter — with it. Slots outside
    this articulation's DOF range are left untouched. Placement arguments
    (``dof_offset``, ``num_envs``, ``num_joints``, ``env_stride``) match
    :func:`scatter_gain_kernel`.
    """
    i = wp.tid()
    global_dof = int(indices[i])
    env = global_dof // env_stride
    local_dof = global_dof - env * env_stride - dof_offset
    if env >= num_envs or local_dof < 0 or local_dof >= num_joints:
        return
    e_pos = env_id_pos[env]
    j_pos = joint_id_pos[local_dof]
    if e_pos >= 0 and j_pos >= 0:
        dst[i] = values[e_pos, j_pos]


# ---------------------------------------------------------------------------
# Articulation-level kernels: in-graph post-actuator hook.
# ---------------------------------------------------------------------------


@wp.kernel(enable_backward=False)
def sync_torque_telemetry(
    joint_pos_backend: wp.array2d(dtype=wp.float32),
    joint_vel_backend: wp.array2d(dtype=wp.float32),
    joint_pos_target: wp.array2d(dtype=wp.float32),
    joint_vel_target: wp.array2d(dtype=wp.float32),
    joint_stiffness: wp.array2d(dtype=wp.float32),
    joint_damping: wp.array2d(dtype=wp.float32),
    effort_limit: wp.array2d(dtype=wp.float32),
    joint_modes: wp.array(dtype=wp.int32),
    sim_bind_joint_effort: wp.array2d(dtype=wp.float32),
    actuator_computed_effort: wp.array2d(dtype=wp.float32),
    user_to_backend: wp.array(dtype=wp.int32),
    sim_buffers_are_backend_order: bool,
    computed: wp.array2d(dtype=wp.float32),
    applied: wp.array2d(dtype=wp.float32),
):
    """In-graph post-actuator hook: fill ``computed`` / ``applied`` torque telemetry.

    For implicit DOFs we compute the shadow PD locally (no Newton actuator
    runs on these); for explicit DOFs we read the pre-clamp effort the
    actuators just scatter-added into ``actuator_computed_effort`` and the
    post-clamp effort already in ``sim_bind_joint_effort`` (= ``joint_f``).
    When the sim-bound buffers are backend-order, the live joint state
    (``joint_pos_backend`` / ``joint_vel_backend``) and both effort buffers are
    gathered through ``user_to_backend`` so every read resolves to public joint
    ``user_j``; the user-facing targets, gains, limits, and telemetry outputs are
    already user-order and are indexed at ``[i, user_j]`` directly.

    Note: ``effort_limit`` clamps only the PD shadow used for implicit-DOF
    telemetry; the FF written into ``joint_f`` is not bounded by it.
    """
    i, user_j = wp.tid()
    backend_j = user_j
    if sim_buffers_are_backend_order:
        backend_j = user_to_backend[user_j]
    if joint_modes[user_j] == 1:
        err_p = joint_pos_target[i, user_j] - joint_pos_backend[i, backend_j]
        err_v = joint_vel_target[i, user_j] - joint_vel_backend[i, backend_j]
        pd = joint_stiffness[i, user_j] * err_p + joint_damping[i, user_j] * err_v
        limit = effort_limit[i, user_j]
        pd_clipped = wp.clamp(pd, -limit, limit)
        total = pd_clipped + sim_bind_joint_effort[i, backend_j]
        computed[i, user_j] = total
        applied[i, user_j] = total
    else:
        computed[i, user_j] = actuator_computed_effort[i, backend_j]
        applied[i, user_j] = sim_bind_joint_effort[i, backend_j]


def build_implicit_dof_mask(
    actuators: dict[str, ActuatorBase],
    num_joints: int,
    device: str,
) -> tuple[wp.array, torch.Tensor]:
    """Per-DOF mask consumed by :func:`sync_torque_telemetry`.

    Entry is ``1`` for DOFs covered by an
    :class:`~isaaclab.actuators.ImplicitActuator` group, ``0`` otherwise.

    Returns:
        Tuple of ``(wp_mask, torch_owner)``. ``wp_mask`` is the Warp
        view used by the kernel; ``torch_owner`` is the underlying
        :class:`torch.Tensor` whose GPU memory ``wp_mask`` aliases. The
        caller **must keep a reference to** ``torch_owner`` for the
        Warp view's lifetime — otherwise the torch refcount drops to
        zero, the memory becomes eligible for reallocation by the
        caching allocator, and any captured CUDA graph that baked in
        ``wp_mask``'s device pointer will read garbage at replay time.
    """
    modes = torch.zeros(num_joints, dtype=torch.int32, device=device)
    for actuator in actuators.values():
        if not isinstance(actuator, ImplicitActuator):
            continue
        j_ids = actuator.joint_indices
        if isinstance(j_ids, slice) or j_ids is None:
            modes[:] = 1
        else:
            modes[j_ids.long()] = 1
    return wp.from_torch(modes, dtype=wp.int32), modes
