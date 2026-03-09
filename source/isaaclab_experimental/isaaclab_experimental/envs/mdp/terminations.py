# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate terminations (experimental).

This module is intentionally minimal: it only contains termination terms that are currently
used by the experimental manager-based Cartpole task.

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


@wp.kernel
def _joint_pos_out_of_manual_limit_kernel(
    joint_pos: wp.array(dtype=wp.float32, ndim=2),
    joint_mask: wp.array(dtype=wp.bool),
    lower: float,
    upper: float,
    out: wp.array(dtype=wp.bool),
):
    i = wp.tid()
    violated = bool(False)
    for j in range(joint_pos.shape[1]):
        if joint_mask[j]:
            v = joint_pos[i, j]
            if v < lower or v > upper:
                violated = True
                break
    out[i] = violated


def joint_pos_out_of_manual_limit(
    env: ManagerBasedRLEnv, out, bounds: tuple[float, float], asset_cfg: SceneEntityCfg
) -> None:
    """Terminate when joint positions are outside configured bounds. Writes into ``out``."""
    asset: Articulation = env.scene[asset_cfg.name]
    assert asset_cfg.joint_mask is not None
    assert asset.data.joint_pos.shape[1] == asset_cfg.joint_mask.shape[0]
    wp.launch(
        kernel=_joint_pos_out_of_manual_limit_kernel,
        dim=env.num_envs,
        inputs=[asset.data.joint_pos, asset_cfg.joint_mask, bounds[0], bounds[1], out],
        device=env.device,
    )
