# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared Warp kernels for the Newton actuator fast path."""

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
    src: wp.array(dtype=wp.float32),
    dst: wp.array(dtype=wp.float32),
    indices: wp.array(dtype=wp.uint32),
    dof_offset: int,
    num_joints: int,
):
    """Scatter per-actuator ``src`` values into a flat per-env-per-DOF ``dst``.

    Used at adapter finalize to snapshot each ``controller.kp`` /
    ``controller.kd`` into the ``(num_envs, num_joints)`` torch tensor
    that ``randomize_actuator_gains`` reads as
    ``actuator.stiffness`` / ``.damping`` for its
    ``default_joint_stiffness`` / ``default_joint_damping`` baseline.
    """
    i = wp.tid()
    global_dof = int(indices[i]) - dof_offset
    env = global_dof // num_joints
    local_dof = global_dof % num_joints
    dst[env * num_joints + local_dof] = src[i]


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
    joint_pos: wp.array2d(dtype=wp.float32),
    joint_vel: wp.array2d(dtype=wp.float32),
    joint_pos_target: wp.array2d(dtype=wp.float32),
    joint_vel_target: wp.array2d(dtype=wp.float32),
    joint_stiffness: wp.array2d(dtype=wp.float32),
    joint_damping: wp.array2d(dtype=wp.float32),
    effort_limit: wp.array2d(dtype=wp.float32),
    joint_modes: wp.array(dtype=wp.int32),
    sim_bind_joint_effort: wp.array2d(dtype=wp.float32),
    actuator_computed_effort: wp.array2d(dtype=wp.float32),
    computed: wp.array2d(dtype=wp.float32),
    applied: wp.array2d(dtype=wp.float32),
):
    """In-graph post-actuator hook: fill ``computed`` / ``applied`` torque telemetry.

    For implicit DOFs we compute the shadow PD locally (no Newton actuator
    runs on these); for explicit DOFs we read the pre-clamp effort the
    actuators just scatter-added into ``actuator_computed_effort`` and the
    post-clamp effort already in ``sim_bind_joint_effort`` (= ``joint_f``).

    Note: ``effort_limit`` clamps only the PD shadow used for implicit-DOF
    telemetry; the FF written into ``joint_f`` is not bounded by it.
    """
    i, j = wp.tid()
    if joint_modes[j] == 1:
        err_p = joint_pos_target[i, j] - joint_pos[i, j]
        err_v = joint_vel_target[i, j] - joint_vel[i, j]
        pd = joint_stiffness[i, j] * err_p + joint_damping[i, j] * err_v
        limit = effort_limit[i, j]
        pd_clipped = wp.clamp(pd, -limit, limit)
        total = pd_clipped + sim_bind_joint_effort[i, j]
        computed[i, j] = total
        applied[i, j] = total
    else:
        computed[i, j] = actuator_computed_effort[i, j]
        applied[i, j] = sim_bind_joint_effort[i, j]


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
