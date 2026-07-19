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
# and a partial scatter for DR gain updates that overwrites only the cells
# in a (env_ids × joint_ids) sub-grid of a Newton ``Actuator``'s controller
# parameter array. Used on the PhysX backend (no Newton view available);
# the Newton backend uses ``ArticulationView.set_actuator_parameter`` instead.
# ---------------------------------------------------------------------------


@wp.kernel(enable_backward=False)
def fill_at_indices_kernel(dst: wp.array(dtype=Any), indices: wp.array(dtype=Any), value: Any):
    """Write ``value`` into a flat ``dst`` buffer at the given flat ``indices``.

    The one fill-shaped scatter: zeroing effort slots, setting mask bits, and
    marking managed DOFs are all instances (see the overloads below).
    """
    i = wp.tid()
    dst[indices[i]] = value


wp.overload(
    fill_at_indices_kernel,
    {"dst": wp.array(dtype=wp.float32), "indices": wp.array(dtype=wp.uint32), "value": wp.float32},
)
wp.overload(
    fill_at_indices_kernel, {"dst": wp.array(dtype=wp.bool), "indices": wp.array(dtype=wp.int32), "value": wp.bool}
)
wp.overload(
    fill_at_indices_kernel, {"dst": wp.array(dtype=wp.bool), "indices": wp.array(dtype=wp.uint32), "value": wp.bool}
)


@wp.kernel(enable_backward=False)
def scatter_flat_kernel(
    src: wp.array(dtype=wp.float32),
    indices: wp.array(dtype=wp.uint32),
    dst: wp.array(dtype=wp.float32),
):
    """Scatter per-actuator values into a flat global-DOF buffer: ``dst[indices[i]] = src[i]``."""
    i = wp.tid()
    dst[indices[i]] = src[i]


@wp.kernel(enable_backward=False)
def gather_env_mask_kernel(
    indices: wp.array(dtype=wp.uint32),
    env_mask: wp.array(dtype=wp.bool),
    dof_env_id: wp.array(dtype=wp.int32),
    out_mask: wp.array(dtype=wp.bool),
):
    """Map a per-env mask onto one actuator's per-DOF layout via the flat DOF->env table.

    Replaces stride arithmetic: works for any flat layout, homogeneous or
    heterogeneous. DOFs outside any environment (``dof_env_id < 0``) never match.
    """
    i = wp.tid()
    env = dof_env_id[wp.int32(indices[i])]
    if env >= 0:
        out_mask[i] = env_mask[env]
    else:
        out_mask[i] = False


@wp.kernel(enable_backward=False)
def patch_actuator_param_kernel(
    indices: wp.array(dtype=wp.uint32),
    env_id_pos: wp.array(dtype=wp.int32),
    joint_id_pos: wp.array(dtype=wp.int32),
    values: wp.array2d(dtype=wp.float32),
    dof_offset: int,
    num_joints: int,
    dst: wp.array(dtype=wp.float32),
):
    """Per-actuator scatter for partial DR gain updates.

    For each slot ``i`` in the actuator's flat env-major ``indices``, derive
    the (env, local-joint) pair, look it up against the dense position
    arrays, and — when both axes are in the DR sub-grid — overwrite
    ``dst[i]`` (the controller parameter) with ``values[e_pos, j_pos]``.
    Cells outside the sub-grid are left untouched.

    Note:
        This kernel is PhysX-only (the Newton backend patches gains via
        :meth:`ArticulationView.set_actuator_parameter`). On PhysX every
        joint's coordinate count equals its DOF count, so the per-env stride
        used to build ``indices`` equals ``num_joints`` and the ``env`` /
        ``joint`` split below is exact. Do not reuse this kernel on a layout
        whose per-env DOF stride exceeds ``num_joints`` (e.g. a floating-base
        Newton model) without threading the true stride, or the ``joint``
        split will alias across envs — see :func:`scatter_gain_kernel`.

    Args:
        indices: Actuator's flat indices into the (env-major) DOF layout.
        env_id_pos: ``env_id_pos[env]`` gives the row in ``values`` for
            envs being updated, ``-1`` otherwise. Length ``num_envs``.
        joint_id_pos: ``joint_id_pos[joint]`` gives the column in
            ``values`` for joints being updated, ``-1`` otherwise.
            Length ``num_joints`` (articulation-local).
        values: New parameter values shaped ``(len(env_ids), len(joint_ids))``.
        dof_offset: Offset of this articulation's DOFs in the env-major
            global index space (``0`` on PhysX, view-dependent on Newton).
        num_joints: Articulation-local joint count.
        dst: Per-actuator controller parameter array (e.g. ``controller.kp``).
    """
    i = wp.tid()
    global_dof = int(indices[i]) - dof_offset
    env = global_dof // num_joints
    joint = global_dof % num_joints
    e_pos = env_id_pos[env]
    j_pos = joint_id_pos[joint]
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
    ``j``; the user-facing targets, gains, limits, and telemetry outputs are
    already user-order and are indexed at ``[i, j]`` directly.

    Note: ``effort_limit`` clamps only the PD shadow used for implicit-DOF
    telemetry; the FF written into ``joint_f`` is not bounded by it.
    """
    i, j = wp.tid()
    backend_j = j
    if sim_buffers_are_backend_order:
        backend_j = user_to_backend[j]
    if joint_modes[j] == 1:
        err_p = joint_pos_target[i, j] - joint_pos_backend[i, backend_j]
        err_v = joint_vel_target[i, j] - joint_vel_backend[i, backend_j]
        pd = joint_stiffness[i, j] * err_p + joint_damping[i, j] * err_v
        limit = effort_limit[i, j]
        pd_clipped = wp.clamp(pd, -limit, limit)
        total = pd_clipped + sim_bind_joint_effort[i, backend_j]
        computed[i, j] = total
        applied[i, j] = total
    else:
        computed[i, j] = actuator_computed_effort[i, backend_j]
        applied[i, j] = sim_bind_joint_effort[i, backend_j]


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
        if j_ids == slice(None) or j_ids is None:
            modes[:] = 1
        else:
            modes[j_ids.long()] = 1
    return wp.from_torch(modes, dtype=wp.int32), modes
