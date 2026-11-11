# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Event terms, the MDP2 counterparts of :mod:`isaaclab.envs.mdp.events`.

Events write their ``outputs`` for the selected worlds. Ranges are ``(low, high)`` pairs; root ranges are six pairs in
the order x, y, z, roll, pitch, yaw, where MDP1 takes a dict. Each event draws from its use's random stream, the same
draws on both backends.
"""

from __future__ import annotations

import math

import torch
import warp as wp

from isaaclab.utils import math as math_utils

from ..rng import uniform
from ..terms import Term

# Joint state.


@wp.func
def _reset_joints_by_offset(
    world: int,
    joint_pos: wp.array2d(dtype=float),
    joint_vel: wp.array2d(dtype=float),
    default_joint_pos: wp.array2d(dtype=float),
    default_joint_vel: wp.array2d(dtype=float),
    soft_joint_pos_limits: wp.array2d(dtype=wp.vec2f),
    soft_joint_vel_limits: wp.array2d(dtype=float),
    joint_ids: wp.array(dtype=wp.int32),
    position_range: wp.vec2f,
    velocity_range: wp.vec2f,
    rng: wp.int32,
):
    num_joints = joint_ids.shape[0]
    for k in range(num_joints):
        j = joint_ids[k]
        position_sample = wp.randf(wp.rand_init(rng, k), position_range[0], position_range[1])
        velocity_sample = wp.randf(wp.rand_init(rng, num_joints + k), velocity_range[0], velocity_range[1])
        position = default_joint_pos[world, j] + position_sample
        velocity = default_joint_vel[world, j] + velocity_sample

        limits = soft_joint_pos_limits[world, j]
        velocity_limit = soft_joint_vel_limits[world, j]
        joint_pos[world, j] = wp.clamp(position, limits[0], limits[1])
        joint_vel[world, j] = wp.clamp(velocity, -velocity_limit, velocity_limit)


@wp.func
def _reset_joints_by_scale(
    world: int,
    joint_pos: wp.array2d(dtype=float),
    joint_vel: wp.array2d(dtype=float),
    default_joint_pos: wp.array2d(dtype=float),
    default_joint_vel: wp.array2d(dtype=float),
    soft_joint_pos_limits: wp.array2d(dtype=wp.vec2f),
    soft_joint_vel_limits: wp.array2d(dtype=float),
    joint_ids: wp.array(dtype=wp.int32),
    position_range: wp.vec2f,
    velocity_range: wp.vec2f,
    rng: wp.int32,
):
    num_joints = joint_ids.shape[0]
    for k in range(num_joints):
        j = joint_ids[k]
        position_sample = wp.randf(wp.rand_init(rng, k), position_range[0], position_range[1])
        velocity_sample = wp.randf(wp.rand_init(rng, num_joints + k), velocity_range[0], velocity_range[1])
        position = default_joint_pos[world, j] * position_sample
        velocity = default_joint_vel[world, j] * velocity_sample

        limits = soft_joint_pos_limits[world, j]
        velocity_limit = soft_joint_vel_limits[world, j]
        joint_pos[world, j] = wp.clamp(position, limits[0], limits[1])
        joint_vel[world, j] = wp.clamp(velocity, -velocity_limit, velocity_limit)


def _joint_samples(rng, joint_ids, position_range, velocity_range):
    """Uniform position and velocity samples per world and joint, at the draws of the Warp funcs."""
    num_joints = len(joint_ids)
    position = uniform(rng, *position_range, (num_joints,))
    velocity = uniform(rng, *velocity_range, (num_joints,), offset=num_joints)
    return position, velocity


def _write_joint_state(
    mask, joint_pos, joint_vel, position, velocity, soft_joint_pos_limits, soft_joint_vel_limits, joint_ids
):
    """Write a joint state, clamped to the soft limits, to the masked worlds."""
    limits = soft_joint_pos_limits[:, joint_ids]
    velocity_limits = soft_joint_vel_limits[:, joint_ids]
    position = position.clamp(limits[..., 0], limits[..., 1])
    velocity = velocity.clamp(-velocity_limits, velocity_limits)

    joint_pos[:, joint_ids] = torch.where(mask[:, None], position, joint_pos[:, joint_ids])
    joint_vel[:, joint_ids] = torch.where(mask[:, None], velocity, joint_vel[:, joint_ids])


def _reset_joints_by_offset_torch(
    mask,
    joint_pos,
    joint_vel,
    default_joint_pos,
    default_joint_vel,
    soft_joint_pos_limits,
    soft_joint_vel_limits,
    joint_ids,
    rng,
    *,
    position_range,
    velocity_range,
):
    position, velocity = _joint_samples(rng, joint_ids, position_range, velocity_range)
    position = default_joint_pos[:, joint_ids] + position
    velocity = default_joint_vel[:, joint_ids] + velocity
    _write_joint_state(
        mask, joint_pos, joint_vel, position, velocity, soft_joint_pos_limits, soft_joint_vel_limits, joint_ids
    )


def _reset_joints_by_scale_torch(
    mask,
    joint_pos,
    joint_vel,
    default_joint_pos,
    default_joint_vel,
    soft_joint_pos_limits,
    soft_joint_vel_limits,
    joint_ids,
    rng,
    *,
    position_range,
    velocity_range,
):
    position, velocity = _joint_samples(rng, joint_ids, position_range, velocity_range)
    position = default_joint_pos[:, joint_ids] * position
    velocity = default_joint_vel[:, joint_ids] * velocity
    _write_joint_state(
        mask, joint_pos, joint_vel, position, velocity, soft_joint_pos_limits, soft_joint_vel_limits, joint_ids
    )


reset_joints_by_offset = Term(torch=_reset_joints_by_offset_torch, warp=_reset_joints_by_offset, random=True)
"""Set the joint state to the default plus uniform offsets from ``position_range`` and ``velocity_range``."""

reset_joints_by_scale = Term(torch=_reset_joints_by_scale_torch, warp=_reset_joints_by_scale, random=True)
"""Set the joint state to the default scaled by uniform factors from ``position_range`` and ``velocity_range``."""


# Root state.


@wp.func
def _uniform_spatial(
    rng: wp.int32, offset: int, ranges: wp.types.matrix(shape=(6, 2), dtype=wp.float32)
) -> wp.spatial_vectorf:
    """Return six uniform samples, the ``j``-th from ``ranges[j]`` at draw ``offset + j``."""
    sample = wp.spatial_vectorf()
    for j in range(6):
        sample[j] = wp.randf(wp.rand_init(rng, offset + j), ranges[j, 0], ranges[j, 1])
    return sample


def _uniform_spatial_torch(rng: torch.Tensor, offset: int, ranges) -> torch.Tensor:
    """Return six uniform samples per world, the ``j``-th from ``ranges[j]`` at draw ``offset + j``."""
    return torch.stack([uniform(rng, low, high, offset=offset + j) for j, (low, high) in enumerate(ranges)], dim=-1)


@wp.func
def _reset_root_state_uniform(
    world: int,
    root_link_pose_w: wp.array(dtype=wp.transformf),
    root_com_vel_w: wp.array(dtype=wp.spatial_vectorf),
    default_root_pose: wp.array(dtype=wp.transformf),
    default_root_vel: wp.array(dtype=wp.spatial_vectorf),
    env_origins: wp.array(dtype=wp.vec3f),
    pose_range: wp.types.matrix(shape=(6, 2), dtype=wp.float32),
    velocity_range: wp.types.matrix(shape=(6, 2), dtype=wp.float32),
    rng: wp.int32,
):
    pose = _uniform_spatial(rng, 0, pose_range)
    default_pose = default_root_pose[world]

    position = wp.transform_get_translation(default_pose) + env_origins[world] + wp.spatial_top(pose)
    delta = wp.quat_rpy(pose[3], pose[4], pose[5])
    orientation = wp.transform_get_rotation(default_pose) * delta
    velocity = default_root_vel[world] + _uniform_spatial(rng, 6, velocity_range)

    root_link_pose_w[world] = wp.transformf(position, orientation)
    root_com_vel_w[world] = velocity


def _reset_root_state_uniform_torch(
    mask,
    root_link_pose_w,
    root_com_vel_w,
    default_root_pose,
    default_root_vel,
    env_origins,
    rng,
    *,
    pose_range,
    velocity_range,
):
    pose = _uniform_spatial_torch(rng, 0, pose_range)

    position = default_root_pose[:, :3] + env_origins + pose[:, :3]
    delta = math_utils.quat_from_euler_xyz(pose[:, 3], pose[:, 4], pose[:, 5])
    orientation = math_utils.quat_mul(default_root_pose[:, 3:7], delta)
    velocity = default_root_vel + _uniform_spatial_torch(rng, 6, velocity_range)

    root_link_pose_w.copy_(torch.where(mask[:, None], torch.cat([position, orientation], -1), root_link_pose_w))
    root_com_vel_w.copy_(torch.where(mask[:, None], velocity, root_com_vel_w))


reset_root_state_uniform = Term(torch=_reset_root_state_uniform_torch, warp=_reset_root_state_uniform, random=True)
"""Set the root pose and velocity to their defaults plus uniform samples from ``pose_range`` and ``velocity_range``.

The pose is relative to the environment origin. Floating bases only.
"""


@wp.func
def _push_by_setting_velocity(
    world: int,
    root_com_vel_w: wp.array(dtype=wp.spatial_vectorf),
    velocity_range: wp.types.matrix(shape=(6, 2), dtype=wp.float32),
    rng: wp.int32,
):
    root_com_vel_w[world] = root_com_vel_w[world] + _uniform_spatial(rng, 0, velocity_range)


def _push_by_setting_velocity_torch(mask, root_com_vel_w, rng, *, velocity_range):
    velocity = root_com_vel_w + _uniform_spatial_torch(rng, 0, velocity_range)
    root_com_vel_w.copy_(torch.where(mask[:, None], velocity, root_com_vel_w))


push_by_setting_velocity = Term(torch=_push_by_setting_velocity_torch, warp=_push_by_setting_velocity, random=True)
"""Add uniform samples from ``velocity_range`` to the root velocity. Floating bases only."""


# Physical properties. These events change the physics model, so they are eager and best run at startup.


def _randomize_rigid_body_mass(
    ctx, *, mass_distribution_params, operation, distribution="uniform", recompute_inertia=True
):
    asset = ctx.entity("body_ids")
    body_ids = ctx.inputs["body_ids"].long()
    env_ids = ctx.mask.nonzero().squeeze(-1)
    rows = env_ids[:, None]

    low, high = mass_distribution_params
    if distribution == "log_uniform":
        samples = uniform(ctx.rng, math.log(low), math.log(high), (len(body_ids),)).exp()[env_ids]
    else:
        samples = uniform(ctx.rng, low, high, (len(body_ids),))[env_ids]

    masses = asset.data.body_mass.torch[rows, body_ids]
    if operation == "add":
        new_masses = masses + samples
    elif operation == "scale":
        new_masses = masses * samples
    else:
        new_masses = samples
    new_masses = new_masses.clamp(min=1e-6)
    asset.set_masses_index(masses=new_masses, body_ids=body_ids, env_ids=env_ids)

    # Assume uniform density: the inertia scales with the mass.
    if recompute_inertia:
        inertias = asset.data.body_inertia.torch[rows, body_ids] * (new_masses / masses)[..., None]
        asset.set_inertias_index(inertias=inertias, body_ids=body_ids, env_ids=env_ids)


randomize_rigid_body_mass = Term(eager=_randomize_rigid_body_mass, random=True)
"""Add to, scale, or set (``operation`` ``"add"``, ``"scale"``, ``"abs"``) the masses of the ``body_ids`` bodies.

Samples come from ``mass_distribution_params``, ``"uniform"`` or ``"log_uniform"``. With ``recompute_inertia``, the
inertias scale with the masses.
"""


def _randomize_rigid_body_com(ctx, *, com_range):
    asset = ctx.entity("body_ids")
    body_ids = ctx.inputs["body_ids"].long()
    env_ids = ctx.mask.nonzero().squeeze(-1)

    offset = torch.stack([uniform(ctx.rng, low, high, offset=axis) for axis, (low, high) in enumerate(com_range)], -1)
    coms = asset.data.body_com_pos_b.torch[env_ids[:, None], body_ids] + offset[env_ids, None]
    asset.set_coms_index(coms=coms, body_ids=body_ids, env_ids=env_ids)


randomize_rigid_body_com = Term(eager=_randomize_rigid_body_com, random=True)
"""Offset the centers of mass of the ``body_ids`` bodies by one uniform sample per world from ``com_range``.

``com_range`` holds three ``(low, high)`` pairs in the order x, y, z, where MDP1 takes a dict.
"""


def _randomize_rigid_body_material(ctx, *, friction_range, restitution_range):
    from newton import ModelFlags  # noqa: PLC0415

    from isaaclab.sim import SimulationContext  # noqa: PLC0415

    asset = ctx.entity("body_ids")
    manager = SimulationContext.instance().backend_manager
    if manager.shares_materials:
        raise ValueError("randomize_rigid_body_material does not support solvers that share materials across worlds.")
    env_ids = ctx.mask.nonzero().squeeze(-1)

    # The view orders shapes as the model does, not by body, so gather the shapes of each selected body.
    body_shapes = asset._root_view.body_shapes
    body_ids = asset.map_body_ids_to_backend(ctx.inputs["body_ids"].tolist())
    shape_ids = torch.tensor([shape for body in body_ids for shape in body_shapes[body]], device=ctx.mask.device)

    model = manager.backend.model
    friction = wp.to_torch(asset._root_view.get_attribute("shape_material_mu", model)[:, 0])
    restitution = wp.to_torch(asset._root_view.get_attribute("shape_material_restitution", model)[:, 0])
    num_shapes = len(shape_ids)
    friction[env_ids[:, None], shape_ids] = uniform(ctx.rng, *friction_range, (num_shapes,))[env_ids]
    restitution_samples = uniform(ctx.rng, *restitution_range, (num_shapes,), offset=num_shapes)
    restitution[env_ids[:, None], shape_ids] = restitution_samples[env_ids]
    manager.add_model_change(ModelFlags.SHAPE_PROPERTIES)


randomize_rigid_body_material = Term(eager=_randomize_rigid_body_material, random=True)
"""Draw the friction and restitution of the shapes of the ``body_ids`` bodies uniformly, per world and shape.

Articulations on Newton only. Newton has one friction coefficient, so there is no separate dynamic friction. Solvers
that share materials across worlds, like Kamino's own collision detection, are not supported.
"""
