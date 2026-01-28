# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import warp as wp
from isaaclab_experimental.managers import SceneEntityCfg
from isaaclab_experimental.utils.warp.utils import wrap_to_pi

from isaaclab.assets import Articulation

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


@wp.kernel
def _joint_pos_target_l2_kernel(
    joint_pos: wp.array(dtype=wp.float32, ndim=2),
    joint_mask: wp.array(dtype=wp.bool),
    out: wp.array(dtype=wp.float32),
    target: float,
):
    i = wp.tid()
    s = float(0.0)
    for j in range(joint_pos.shape[1]):
        if joint_mask[j]:
            a = wrap_to_pi(joint_pos[i, j])
            d = a - target
            s += d * d
    out[i] = s


def joint_pos_target_l2(env: ManagerBasedRLEnv, out, target: float, asset_cfg: SceneEntityCfg) -> None:
    """Penalize joint position deviation from a target value. Writes into ``out``."""
    asset: Articulation = env.scene[asset_cfg.name]
    assert asset.data.joint_pos.shape[1] == asset_cfg.joint_mask.shape[0]
    wp.launch(
        kernel=_joint_pos_target_l2_kernel,
        dim=env.num_envs,
        inputs=[asset.data.joint_pos, asset_cfg.joint_mask, out, target],
        device=env.device,
    )
