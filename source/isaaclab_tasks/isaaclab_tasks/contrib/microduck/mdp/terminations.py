# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Numerical-state and recovery-timeout termination terms."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv

from isaaclab.managers import ManagerTermBase, TerminationTermCfg

from .rewards import _is_fallen


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
        net_forces_w = env.scene.sensors[name].data.net_forces_w.torch
        is_broken |= ~torch.isfinite(net_forces_w).flatten(start_dim=1).all(dim=1)
    return is_broken


def robot_state_is_nan_optional_sensors(
    env: ManagerBasedRLEnv,
    sensor_names: Sequence[str] = (),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate environments whose robot state has stopped being finite.

    This is not a time-out: configure it with ``time_out=False`` so the value bootstraps as a real
    failure.

    There is no stock counterpart. :func:`isaaclab.envs.mdp.root_height_below_minimum` and
    :func:`~isaaclab.envs.mdp.bad_orientation` catch a fallen robot, not a broken one, and a NaN
    height compares false against every bound rather than tripping them.

    Args:
        env: The environment instance.
        sensor_names: Names of contact sensors whose forces are checked as well. Names the scene
            does not carry are ignored, as does. Defaults to an empty tuple.
        asset_cfg: The articulation to check. Defaults to the entity named ``"robot"``.

    Returns:
        Whether each environment must terminate. Shape is (num_envs,).
    """
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


class fallen_too_long(ManagerTermBase):
    """Terminate an environment whose robot has been continuously down for too long.

    Two properties are deliberate and both differ from the recovery *rewards*:

    This is not a time-out: configure it with ``time_out=False`` so the value bootstraps as a real
    failure, which is what makes lying down expensive to the critic.

    Being a clock the term is stateful, so it is a class.
    """

    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRLEnv):
        """Allocate the per-environment fallen clock.

        Args:
            cfg: The term configuration.
            env: The environment instance.
        """
        super().__init__(cfg, env)

        self._fallen_s = torch.zeros(env.num_envs, device=env.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Clear the clock for the environments that restarted.

        A new episode that spawns prone gets the whole timeout, not the remainder of the previous
        episode's.

        Args:
            env_ids: The environment ids. Defaults to None, in which case all are cleared.
        """
        if env_ids is None:
            self._fallen_s[:] = 0.0
        else:
            self._fallen_s[env_ids] = 0.0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        gate_z_below: float,
        gate_tilt_above_deg: float,
        max_duration_s: float,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        """Advance the clock and report the environments that have run out of it.

        Args:
            env: The environment instance.
            gate_z_below: Trunk height [m] below which the robot counts as down.
            gate_tilt_above_deg: Trunk tilt [deg] beyond which it counts as down, whatever its height.
            max_duration_s: Continuous time [s] spent down before the episode is recycled.
            asset_cfg: The articulation whose root link carries the trunk.

        Returns:
            Whether each environment must terminate. Shape is (num_envs,).
        """
        asset: Articulation = env.scene[asset_cfg.name]
        fallen = _is_fallen(env, asset, gate_tilt_above_deg, gate_z_below)
        self._fallen_s = torch.where(fallen, self._fallen_s + env.step_dt, torch.zeros_like(self._fallen_s))
        return self._fallen_s >= max_duration_s
