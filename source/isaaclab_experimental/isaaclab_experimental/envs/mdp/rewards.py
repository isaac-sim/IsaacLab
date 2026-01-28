# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to enable reward functions (experimental).

This module is intentionally minimal: it only contains reward terms that are currently
used by the experimental manager-based Cartpole task.

All functions in this file follow the Warp-compatible reward signature expected by
`isaaclab_experimental.managers.RewardManager`:

- ``func(env, out, **params) -> None``

where ``out`` is a pre-allocated Warp array of shape ``(num_envs,)`` with ``float32`` dtype.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import warp as wp
from isaaclab_experimental.managers import SceneEntityCfg

from isaaclab.assets import Articulation

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


"""
General.
"""


@wp.kernel
def _is_alive_kernel(terminated: wp.array(dtype=wp.bool), out: wp.array(dtype=wp.float32)):
    i = wp.tid()
    out[i] = wp.where(terminated[i], 0.0, 1.0)


def is_alive(env: ManagerBasedRLEnv, out: wp.array(dtype=wp.float32)) -> None:
    """Reward for being alive. Writes into ``out`` (shape: (num_envs,))."""
    terminated_wp = wp.from_torch(env.termination_manager.terminated, dtype=wp.bool)
    wp.launch(kernel=_is_alive_kernel, dim=env.num_envs, inputs=[terminated_wp, out], device=env.device)


@wp.kernel
def _is_terminated_kernel(terminated: wp.array(dtype=wp.bool), out: wp.array(dtype=wp.float32)):
    i = wp.tid()
    out[i] = wp.where(terminated[i], 1.0, 0.0)


def is_terminated(env: ManagerBasedRLEnv, out) -> None:
    """Penalize terminated episodes. Writes into ``out``."""
    terminated_wp = wp.from_torch(env.termination_manager.terminated, dtype=wp.bool)
    wp.launch(kernel=_is_terminated_kernel, dim=env.num_envs, inputs=[terminated_wp, out], device=env.device)


"""
Joint penalties.
"""


@wp.kernel
def _sum_abs_masked_kernel(
    x: wp.array(dtype=wp.float32, ndim=2), joint_mask: wp.array(dtype=wp.bool), out: wp.array(dtype=wp.float32)
):
    i = wp.tid()
    s = float(0.0)
    for j in range(x.shape[1]):
        if joint_mask[j]:
            s += wp.abs(x[i, j])
    out[i] = s


def joint_vel_l1(env: ManagerBasedRLEnv, out, asset_cfg: SceneEntityCfg) -> None:
    """Penalize joint velocities on the articulation using an L1-kernel. Writes into ``out``."""
    asset: Articulation = env.scene[asset_cfg.name]
    wp.launch(
        kernel=_sum_abs_masked_kernel,
        dim=env.num_envs,
        inputs=[asset.data.joint_vel, asset_cfg.joint_mask, out],
        device=env.device,
    )
