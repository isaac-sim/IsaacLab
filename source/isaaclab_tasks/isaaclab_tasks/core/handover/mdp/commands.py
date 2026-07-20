# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Goal-pose command for the manager-based handover task."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.core.handover.handover_common import GOAL_MARKER_CFG, GOAL_POSITION_OFFSET

if TYPE_CHECKING:
    from isaaclab.assets import RigidObject
    from isaaclab.envs import ManagerBasedRLEnv


class HandoverCommand(CommandTerm):
    """Sample the fixed-position, random-orientation handover goal pose."""

    cfg: HandoverCommandCfg

    def __init__(self, cfg: HandoverCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._object: RigidObject = env.scene[cfg.asset_name]
        offset = torch.tensor(cfg.position_offset, dtype=torch.float, device=self.device)
        self.pos_command_e = self._object.data.default_root_pose.torch[:, :3] + offset
        self.quat_command_w = torch.zeros(self.num_envs, 4, device=self.device)
        self.quat_command_w[:, 3] = 1.0  # identity quaternion in (x, y, z, w) layout
        # persistent (num_envs, 7) pose command: the position half is static and written once
        # here, the quaternion half is refreshed by _resample_command; `command` returns this
        # buffer directly instead of allocating a torch.cat every call
        self._command_buf = torch.cat((self.pos_command_e, self.quat_command_w), dim=-1)
        self._x_unit = torch.tensor([1.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)
        self._y_unit = torch.tensor([0.0, 1.0, 0.0], device=self.device).repeat(self.num_envs, 1)

    @property
    def command(self) -> torch.Tensor:
        """Goal pose in the environment frame [m, unit quaternion].

        The returned tensor is a persistent buffer refreshed in place; consumers that
        store it across steps must copy it.
        """
        return self._command_buf

    def _update_metrics(self) -> None:
        pass

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        random_values = 2.0 * torch.rand((len(env_ids), 2), device=self.device) - 1.0
        self.quat_command_w[env_ids] = math_utils.quat_mul(
            math_utils.quat_from_angle_axis(random_values[:, 0] * torch.pi, self._x_unit[env_ids]),
            math_utils.quat_from_angle_axis(random_values[:, 1] * torch.pi, self._y_unit[env_ids]),
        )
        # keep the persistent pose-command buffer current (position half is static)
        self._command_buf[env_ids, 3:] = self.quat_command_w[env_ids]

    def _update_command(self) -> None:
        pass

    def _set_debug_vis_impl(self, debug_vis: bool) -> None:
        if debug_vis:
            if not hasattr(self, "_goal_visualizer"):
                self._goal_visualizer = VisualizationMarkers(self.cfg.goal_visualizer_cfg)
            self._goal_visualizer.set_visibility(True)
        elif hasattr(self, "_goal_visualizer"):
            self._goal_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event) -> None:
        del event
        self._goal_visualizer.visualize(
            translations=self.pos_command_e + self._env.scene.env_origins,
            orientations=self.quat_command_w,
        )


@configclass
class HandoverCommandCfg(CommandTermCfg):
    """Configuration for :class:`HandoverCommand`."""

    class_type: type[HandoverCommand] = HandoverCommand
    resampling_time_range: tuple[float, float] = (1.0e6, 1.0e6)
    asset_name: str = MISSING
    position_offset: tuple[float, float, float] = GOAL_POSITION_OFFSET
    """Goal-position offset from the object's default position [m]."""
    goal_visualizer_cfg: VisualizationMarkersCfg = GOAL_MARKER_CFG.replace(prim_path="/Visuals/Command/goal_marker")
