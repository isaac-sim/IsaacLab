# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Warp-first observation terms (experimental).

All functions in this file follow the Warp-compatible observation signature expected by the
experimental Warp-first observation manager:

- ``func(env, out, **params) -> None``

where ``out`` is a pre-allocated Warp array with float32 dtype and shape ``(num_envs, D)``.
Output dimension ``D`` is inferred from decorator metadata: ``axes`` for root-state terms,
``out_dim`` for body/command/action/time terms, or ``joint_ids`` count for joint terms.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import warp as wp
from isaaclab_experimental.envs.utils.io_descriptors import (
    generic_io_descriptor_warp,
    record_body_names,
    record_dtype,
    record_joint_names,
    record_joint_pos_offsets,
    record_joint_vel_offsets,
    record_shape,
)
from isaaclab_experimental.managers import SceneEntityCfg
from isaaclab_experimental.utils.warp import warp_capturable

from isaaclab.assets import Articulation

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


# ---------------------------------------------------------------------------
# Shared kernels
# ---------------------------------------------------------------------------


@wp.kernel
def _vec3_to_out3_kernel(
    src: wp.array(dtype=wp.vec3f),
    out: wp.array(dtype=wp.float32, ndim=2),
):
    env_id = wp.tid()
    v = src[env_id]
    out[env_id, 0] = v[0]
    out[env_id, 1] = v[1]
    out[env_id, 2] = v[2]


@wp.kernel
def _joint_gather_kernel(
    src: wp.array(dtype=wp.float32, ndim=2),
    joint_ids: wp.array(dtype=wp.int32),
    out: wp.array(dtype=wp.float32, ndim=2),
):
    env_id, k = wp.tid()
    j = joint_ids[k]
    out[env_id, k] = src[env_id, j]


"""
Root state.
"""


@wp.kernel
def _base_pos_z_kernel(
    root_pos_w: wp.array(dtype=wp.vec3f),
    out: wp.array(dtype=wp.float32, ndim=2),
):
    env_id = wp.tid()
    out[env_id, 0] = root_pos_w[env_id][2]


# Reviewed(jichuanh): good
@generic_io_descriptor_warp(
    units="m", axes=["Z"], observation_type="RootState", on_inspect=[record_shape, record_dtype]
)
def base_pos_z(env: ManagerBasedEnv, out, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> None:
    """Root height in the simulation world frame."""
    asset: Articulation = env.scene[asset_cfg.name]
    wp.launch(
        kernel=_base_pos_z_kernel,
        dim=env.num_envs,
        inputs=[asset.data.root_pos_w, out],
        device=env.device,
    )


# Reviewed(jichuanh): good
@warp_capturable(False)  # accesses root_lin_vel_b → lazy TimestampedWarpBuffer (Tier 2)
@generic_io_descriptor_warp(
    units="m/s", axes=["X", "Y", "Z"], observation_type="RootState", on_inspect=[record_shape, record_dtype]
)
def base_lin_vel(env: ManagerBasedEnv, out, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> None:
    """Root linear velocity in the asset's root frame."""
    asset: Articulation = env.scene[asset_cfg.name]
    wp.launch(
        kernel=_vec3_to_out3_kernel,
        dim=env.num_envs,
        inputs=[asset.data.root_lin_vel_b, out],
        device=env.device,
    )


# Reviewed(jichuanh): good
@warp_capturable(False)  # accesses root_ang_vel_b → lazy TimestampedWarpBuffer (Tier 2)
@generic_io_descriptor_warp(
    units="rad/s", axes=["X", "Y", "Z"], observation_type="RootState", on_inspect=[record_shape, record_dtype]
)
def base_ang_vel(env: ManagerBasedEnv, out, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> None:
    """Root angular velocity in the asset's root frame."""
    asset: Articulation = env.scene[asset_cfg.name]
    wp.launch(
        kernel=_vec3_to_out3_kernel,
        dim=env.num_envs,
        inputs=[asset.data.root_ang_vel_b, out],
        device=env.device,
    )


# Reviewed(jichuanh): good
@warp_capturable(False)  # accesses projected_gravity_b → lazy TimestampedWarpBuffer (Tier 2)
@generic_io_descriptor_warp(
    units="m/s^2", axes=["X", "Y", "Z"], observation_type="RootState", on_inspect=[record_shape, record_dtype]
)
def projected_gravity(env: ManagerBasedEnv, out, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> None:
    """Gravity projection on the asset's root frame."""
    asset: Articulation = env.scene[asset_cfg.name]
    wp.launch(
        kernel=_vec3_to_out3_kernel,
        dim=env.num_envs,
        inputs=[asset.data.projected_gravity_b, out],
        device=env.device,
    )


"""
Joint state.
"""


@generic_io_descriptor_warp(
    observation_type="JointState", on_inspect=[record_joint_names, record_dtype, record_shape], units="rad"
)
def joint_pos(env: ManagerBasedEnv, out, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> None:
    """The joint positions of the asset."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_ids_wp = getattr(asset_cfg, "joint_ids_wp", None)
    if joint_ids_wp is None:
        raise RuntimeError(
            "SceneEntityCfg.joint_ids_wp is required for subset joint observations in Warp-first observations. "
            "Pass `asset_cfg` via term cfg params so it is resolved at manager init."
        )
    wp.launch(
        kernel=_joint_gather_kernel,
        dim=(env.num_envs, out.shape[1]),
        inputs=[asset.data.joint_pos, joint_ids_wp, out],
        device=env.device,
    )


@wp.kernel
def _joint_pos_rel_gather_kernel(
    joint_pos: wp.array(dtype=wp.float32, ndim=2),
    default_joint_pos: wp.array(dtype=wp.float32, ndim=2),
    joint_ids: wp.array(dtype=wp.int32),
    out: wp.array(dtype=wp.float32, ndim=2),
):
    env_id, k = wp.tid()
    j = joint_ids[k]
    out[env_id, k] = joint_pos[env_id, j] - default_joint_pos[env_id, j]


# Reviewed(jichuanh): good
@generic_io_descriptor_warp(
    observation_type="JointState",
    on_inspect=[record_joint_names, record_dtype, record_shape, record_joint_pos_offsets],
    units="rad",
)
def joint_pos_rel(env: ManagerBasedEnv, out, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> None:
    """Joint positions relative to defaults. Writes into ``out``."""
    asset: Articulation = env.scene[asset_cfg.name]

    # Subset selection (requires a pre-resolved Warp joint-id list).
    joint_ids_wp = getattr(asset_cfg, "joint_ids_wp", None)
    if joint_ids_wp is None:
        raise RuntimeError(
            "SceneEntityCfg.joint_ids_wp is required for subset joint observations in Warp-first observations. "
            "Pass `asset_cfg` via term cfg params so it is resolved at manager init."
        )
    wp.launch(
        kernel=_joint_pos_rel_gather_kernel,
        dim=(env.num_envs, out.shape[1]),
        inputs=[asset.data.joint_pos, asset.data.default_joint_pos, joint_ids_wp, out],
        device=env.device,
    )


# Reviewed(jichuanh): logic is different from stable version. Even upper and lower are flipped, stable
#                     logic should work, fix this.
@wp.kernel
def _joint_pos_limit_normalized_kernel(
    joint_pos: wp.array(dtype=wp.float32, ndim=2),
    soft_joint_pos_limits: wp.array(dtype=wp.vec2f, ndim=2),
    joint_ids: wp.array(dtype=wp.int32),
    out: wp.array(dtype=wp.float32, ndim=2),
):
    env_id, k = wp.tid()
    j = joint_ids[k]
    pos = joint_pos[env_id, j]
    lim = soft_joint_pos_limits[env_id, j]
    lower = lim.x
    upper = lim.y
    mid = (lower + upper) * 0.5
    half_range = (upper - lower) * 0.5
    if half_range > 0.0:
        out[env_id, k] = (pos - mid) / half_range
    else:
        out[env_id, k] = 0.0


@generic_io_descriptor_warp(observation_type="JointState", on_inspect=[record_joint_names, record_dtype, record_shape])
def joint_pos_limit_normalized(env: ManagerBasedEnv, out, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> None:
    """The joint positions of the asset normalized with the asset's joint limits."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_ids_wp = getattr(asset_cfg, "joint_ids_wp", None)
    if joint_ids_wp is None:
        raise RuntimeError(
            "SceneEntityCfg.joint_ids_wp is required for subset joint observations in Warp-first observations. "
            "Pass `asset_cfg` via term cfg params so it is resolved at manager init."
        )
    wp.launch(
        kernel=_joint_pos_limit_normalized_kernel,
        dim=(env.num_envs, out.shape[1]),
        inputs=[asset.data.joint_pos, asset.data.soft_joint_pos_limits, joint_ids_wp, out],
        device=env.device,
    )


# Reviewed(jichuanh): good
@generic_io_descriptor_warp(
    observation_type="JointState", on_inspect=[record_joint_names, record_dtype, record_shape], units="rad/s"
)
def joint_vel(env: ManagerBasedEnv, out, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> None:
    """The joint velocities of the asset."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_ids_wp = getattr(asset_cfg, "joint_ids_wp", None)
    if joint_ids_wp is None:
        raise RuntimeError(
            "SceneEntityCfg.joint_ids_wp is required for subset joint observations in Warp-first observations. "
            "Pass `asset_cfg` via term cfg params so it is resolved at manager init."
        )
    wp.launch(
        kernel=_joint_gather_kernel,
        dim=(env.num_envs, out.shape[1]),
        inputs=[asset.data.joint_vel, joint_ids_wp, out],
        device=env.device,
    )


# Reviewed(jichuanh): kernel impl seems duplicate, rel_gather kernel could be shared.
@wp.kernel
def _joint_vel_rel_gather_kernel(
    joint_vel: wp.array(dtype=wp.float32, ndim=2),
    default_joint_vel: wp.array(dtype=wp.float32, ndim=2),
    joint_ids: wp.array(dtype=wp.int32),
    out: wp.array(dtype=wp.float32, ndim=2),
):
    env_id, k = wp.tid()
    j = joint_ids[k]
    out[env_id, k] = joint_vel[env_id, j] - default_joint_vel[env_id, j]


@generic_io_descriptor_warp(
    observation_type="JointState",
    on_inspect=[record_joint_names, record_dtype, record_shape, record_joint_vel_offsets],
    units="rad/s",
)
def joint_vel_rel(env: ManagerBasedEnv, out, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> None:
    """Joint velocities relative to defaults. Writes into ``out``."""
    asset: Articulation = env.scene[asset_cfg.name]

    # Subset selection (requires a pre-resolved Warp joint-id list).
    joint_ids_wp = getattr(asset_cfg, "joint_ids_wp", None)
    if joint_ids_wp is None:
        raise RuntimeError(
            "SceneEntityCfg.joint_ids_wp is required for subset joint observations in Warp-first observations. "
            "Pass `asset_cfg` via term cfg params so it is resolved at manager init."
        )
    wp.launch(
        kernel=_joint_vel_rel_gather_kernel,
        dim=(env.num_envs, out.shape[1]),
        inputs=[asset.data.joint_vel, asset.data.default_joint_vel, joint_ids_wp, out],
        device=env.device,
    )


"""
Actions.
"""
# Reviewed(jichuanh): good


@generic_io_descriptor_warp(out_dim="action", dtype=torch.float32, observation_type="Action", on_inspect=[record_shape])
def last_action(env: ManagerBasedEnv, out, action_name: str | None = None) -> None:
    """The last input action to the environment."""
    # TODO(warp-migration): Cross-manager access (observation → action). Currently works
    #  because experimental ActionManager.action is already a warp array. No from_torch needed.
    if action_name is not None:
        raise NotImplementedError("Named action support is not yet implemented for Warp-first last_action observation.")
    wp.copy(out, env.action_manager.action)


"""
Commands.
"""


# Reviewed(jichuanh): good
@generic_io_descriptor_warp(
    out_dim="command", dtype=torch.float32, observation_type="Command", on_inspect=[record_shape]
)
def generated_commands(env: ManagerBasedEnv, out, command_name: str) -> None:
    """The generated command from the command manager. Writes into ``out``.

    Warp-first override of :func:`isaaclab.envs.mdp.observations.generated_commands`.
    Uses ``wp.from_torch`` to create a zero-copy warp view of the command tensor on first call.
    """
    # TODO(warp-migration): Cross-manager access (observation → command). Replace with direct
    #  warp getter once all managers are guaranteed to be warp-native.
    fn = generated_commands
    if not hasattr(fn, "_cmd_wp") or fn._cmd_name != command_name:
        cmd = env.command_manager.get_command(command_name)
        if isinstance(cmd, wp.array):
            fn._cmd_wp = cmd
        else:
            fn._cmd_wp = wp.from_torch(cmd)
        fn._cmd_name = command_name
    wp.copy(out, fn._cmd_wp)


