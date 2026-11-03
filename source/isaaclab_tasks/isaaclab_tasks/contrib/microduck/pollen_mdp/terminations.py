# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Termination terms MicroDuck needs that have no stock Isaac Lab counterpart.

Ported from ``pollen-robotics/microduck_rl``; see section 6 of
``historical original source extraction`` for the verbatim upstream formula.

Source pins and historical extraction labels are indexed in this task family's ``UPSTREAM.md``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, SceneEntityCfg, TerminationTermCfg

# the fallen test is shared with the recovery rewards, as it is upstream, so that the termination
# and the terms it recycles cannot drift apart on what "fallen" means
from .rewards import _is_fallen

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv


def robot_state_is_nan(
    env: ManagerBasedRLEnv,
    sensor_names: Sequence[str] = (),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate environments whose robot state has stopped being finite.

    Ported from reference section 6 (``robot_state_is_nan``). A solver that diverges leaves NaN or
    infinite values in the state, which then propagate into the observations and poison the policy
    gradient for the whole batch; terminating resets the environment instead. Despite its name the
    check is for non-finiteness, so infinities are caught as well.

    This is not a time-out: configure it with ``time_out=False`` so the value bootstraps as a real
    failure.

    There is no stock counterpart. :func:`isaaclab.envs.mdp.root_height_below_minimum` and
    :func:`~isaaclab.envs.mdp.bad_orientation` catch a fallen robot, not a broken one, and a NaN
    height compares false against every bound rather than tripping them.

    Args:
        env: The environment instance.
        sensor_names: Names of contact sensors whose forces are checked as well. Names the scene
            does not carry are ignored, as upstream does. Defaults to an empty tuple.
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

    Ported from addendum section 3.5 (``fallen_too_long``). It is the backstop of the hybrid
    walking-and-recovery task: once the tilt termination is disabled by curriculum a fall no longer
    ends the episode, and without this term a robot that never gets up would spend the remaining
    eighteen seconds of its episode on the floor, filling the rollout with data about lying down.

    Two properties are deliberate and both differ from the recovery *rewards*:

    * The gate is **height or tilt**, where the rewards gate on tilt alone. That asymmetry is
      upstream's: a robot sitting low but upright is not paid as fallen, but it is recycled as
      stuck, so sitting is neither rewarded nor a comfortable place to wait.
    * The clock measures a **continuous** fall. Getting up at any point clears it, so a robot that
      falls repeatedly is not accumulating a budget across recoveries.

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
