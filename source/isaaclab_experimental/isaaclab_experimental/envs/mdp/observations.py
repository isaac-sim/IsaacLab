# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Warp-first observation terms (experimental, Cartpole-focused).

All functions in this file follow the Warp-compatible observation signature expected by the
experimental Warp-first observation manager:

- ``func(env, out, **params) -> None``

where ``out`` is a pre-allocated Warp array with float32 dtype and shape ``(num_envs, term_dim)``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import warp as wp

from isaaclab.assets import Articulation

from isaaclab_experimental.envs.utils.io_descriptors import (
    generic_io_descriptor_warp,
    record_joint_names,
    record_joint_pos_offsets,
    record_joint_shape,
    record_joint_vel_offsets,
)
from isaaclab_experimental.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


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


@generic_io_descriptor_warp(
    observation_type="JointState",
    on_inspect=[record_joint_names, record_joint_shape, record_joint_pos_offsets],
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
    on_inspect=[record_joint_names, record_joint_shape, record_joint_vel_offsets],
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
