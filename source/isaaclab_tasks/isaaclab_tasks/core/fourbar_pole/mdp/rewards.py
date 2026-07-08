# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Functions specific to the fourbar-pole swing-up environments.

This module holds both observation helpers (``joint_pos_cos`` / ``joint_pos_sin``)
that encode an angle without the ``+-pi`` wrap discontinuity, and the swing-up
reward / success-metric terms.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import CommandTerm, CommandTermCfg, SceneEntityCfg
from isaaclab.utils.configclass import configclass

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv


def joint_pos_cos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Cosine of the selected joint positions.

    Encodes the angle without the wrap-around discontinuity at ``+-pi`` so the
    policy sees a smooth signal as the pole swings through the bottom.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.cos(asset.data.joint_pos.torch[:, asset_cfg.joint_ids])


def joint_pos_sin(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Sine of the selected joint positions (companion to :func:`joint_pos_cos`)."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sin(asset.data.joint_pos.torch[:, asset_cfg.joint_ids])


def pole_upright(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Uprightness reward for the pole, maximal (``+1``) when fully upright.

    The pole joint is zero when upright and ``+-pi`` when hanging, so ``cos`` gives
    a dense shaping signal in ``[-1, 1]`` that drives the swing-up.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.cos(asset.data.joint_pos.torch[:, asset_cfg.joint_ids]), dim=1)


class UprightSuccessRateCommand(CommandTerm):
    """Command term that tracks pole-upright terminal success as a metric."""

    cfg: UprightSuccessRateCommandCfg

    def __init__(self, cfg: UprightSuccessRateCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._asset_cfg = cfg.asset_cfg
        self._asset_cfg.resolve(env.scene)
        self._asset: Articulation = env.scene[self._asset_cfg.name]

        self._command = torch.zeros((self.num_envs, 1), device=self.device)
        self.metrics["success_rate"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self._command

    def _update_metrics(self):
        pole_pos = self._asset.data.joint_pos.torch[:, self._asset_cfg.joint_ids]
        self.metrics["success_rate"] = (torch.cos(pole_pos) > self.cfg.threshold).all(dim=1).float()

    def _resample_command(self, env_ids: Sequence[int]):
        pass

    def _update_command(self):
        pass


@configclass
class UprightSuccessRateCommandCfg(CommandTermCfg):
    """Configuration for :class:`UprightSuccessRateCommand`."""

    class_type: type[UprightSuccessRateCommand] | str = "{DIR}.rewards:UprightSuccessRateCommand"
    resampling_time_range: tuple[float, float] = (1e6, 1e6)

    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["coupler_to_pole"])
    threshold: float = 0.95
