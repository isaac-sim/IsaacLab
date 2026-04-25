# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate terminations (experimental).

All functions in this file follow the Warp-compatible termination signature expected by
`isaaclab_experimental.managers.TerminationManager`:

- ``func(env, out, **params) -> None``

where ``out`` is a pre-allocated Warp array of shape ``(num_envs,)`` with boolean dtype.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import warp as wp

from isaaclab.assets import Articulation

from isaaclab_experimental.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


"""
MDP terminations.
"""


@wp.kernel
def _time_out_kernel(
    episode_length: wp.array(dtype=wp.int64), max_episode_length: wp.int64, out: wp.array(dtype=wp.bool)
):
    i = wp.tid()
    out[i] = episode_length[i] >= max_episode_length


def time_out(env: ManagerBasedRLEnv, out) -> None:
    """Terminate the episode when episode length exceeds the maximum episode length."""
    wp.launch(
        kernel=_time_out_kernel,
        dim=env.num_envs,
        inputs=[env._episode_length_buf_wp, env.max_episode_length, out],
        device=env.device,
    )


"""
Root terminations.
"""


@wp.kernel
def _root_height_below_min_kernel(
    root_pos_w: wp.array(dtype=wp.vec3f),
    minimum_height: float,
    out: wp.array(dtype=wp.bool),
):
    i = wp.tid()
    out[i] = root_pos_w[i][2] < minimum_height


def root_height_below_minimum(
    env: ManagerBasedRLEnv, out, minimum_height: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> None:
    """Terminate when the asset's root height is below the minimum height."""
    asset: Articulation = env.scene[asset_cfg.name]
    wp.launch(
        kernel=_root_height_below_min_kernel,
        dim=env.num_envs,
        inputs=[asset.data.root_pos_w, minimum_height, out],
        device=env.device,
    )


"""
Joint terminations.
"""


@wp.kernel
def _joint_pos_out_of_manual_limit_kernel(
    joint_pos: wp.array(dtype=wp.float32, ndim=2),
    joint_mask: wp.array(dtype=wp.bool),
    lower: float,
    upper: float,
    out: wp.array(dtype=wp.bool),
):
    """2D kernel (num_envs, num_joints). ``out`` is pre-zeroed; only writes True."""
    i, j = wp.tid()
    if joint_mask[j]:
        v = joint_pos[i, j]
        if v < lower or v > upper:
            out[i] = True


def joint_pos_out_of_manual_limit(
    env: ManagerBasedRLEnv, out, bounds: tuple[float, float], asset_cfg: SceneEntityCfg
) -> None:
    """Terminate when joint positions are outside configured bounds. Writes into ``out``."""
    asset: Articulation = env.scene[asset_cfg.name]
    if asset_cfg.joint_mask is None:
        raise ValueError(
            f"joint_pos_out_of_manual_limit requires SceneEntityCfg with resolved joint_mask, "
            f"but got None for asset '{asset_cfg.name}'."
        )
    if asset.data.joint_pos.shape[1] != asset_cfg.joint_mask.shape[0]:
        raise ValueError(
            f"joint_mask length ({asset_cfg.joint_mask.shape[0]}) does not match "
            f"joint_pos dim ({asset.data.joint_pos.shape[1]}) for asset '{asset_cfg.name}'."
        )
    wp.launch(
        kernel=_joint_pos_out_of_manual_limit_kernel,
        dim=(env.num_envs, asset.data.joint_pos.shape[1]),
        inputs=[asset.data.joint_pos, asset_cfg.joint_mask, bounds[0], bounds[1], out],
        device=env.device,
    )


"""
Contact sensor.
"""


@wp.kernel
def _illegal_contact_kernel(
    forces: wp.array(dtype=wp.vec3f, ndim=3),
    body_ids: wp.array(dtype=wp.int32),
    threshold: float,
    out: wp.array(dtype=wp.bool),
):
    """Terminate when any selected body's max-over-history contact force norm exceeds threshold."""
    i = wp.tid()
    violated = bool(False)
    for k in range(body_ids.shape[0]):
        b = body_ids[k]
        for h in range(forces.shape[1]):
            f = forces[i, h, b]
            norm = wp.sqrt(f[0] * f[0] + f[1] * f[1] + f[2] * f[2])
            if norm > threshold:
                violated = True
    out[i] = violated


def illegal_contact(env: ManagerBasedRLEnv, out, threshold: float, sensor_cfg: SceneEntityCfg) -> None:
    """Terminate when the contact force on the sensor exceeds the force threshold. Writes into ``out``.

    Warp-first override of :func:`isaaclab.envs.mdp.terminations.illegal_contact`.
    """
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    wp.launch(
        kernel=_illegal_contact_kernel,
        dim=env.num_envs,
        inputs=[contact_sensor.data.net_forces_w_history, sensor_cfg.body_ids_wp, threshold, out],
        device=env.device,
    )
