# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MicroDuck flat-walking terminations."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv


def robot_state_is_nan(
    env: ManagerBasedRLEnv, sensor_names: Sequence[str] = (), asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terminate environments whose robot state has stopped being finite."""
    asset: Articulation = env.scene[asset_cfg.name]
    data = asset.data
    is_broken = ~torch.isfinite(data.joint_pos.torch).all(dim=1)
    is_broken |= ~torch.isfinite(data.joint_vel.torch).all(dim=1)
    is_broken |= ~torch.isfinite(data.root_link_pos_w.torch).all(dim=1)
    is_broken |= ~torch.isfinite(data.root_link_quat_w.torch).all(dim=1)
    is_broken |= ~torch.isfinite(data.root_link_lin_vel_w.torch).all(dim=1)
    is_broken |= ~torch.isfinite(data.root_link_ang_vel_w.torch).all(dim=1)
    for name in sensor_names:
        if name not in env.scene.sensors:
            continue
        net_forces_w = env.scene.sensors[name].data.net_forces_w
        if net_forces_w is None:
            continue
        is_broken |= ~torch.isfinite(net_forces_w.torch).flatten(start_dim=1).all(dim=1)
    return is_broken
