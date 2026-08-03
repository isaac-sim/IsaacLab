# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pose command terms for the deformable lift tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import AssetBaseCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.math import combine_frame_transforms

from isaaclab_tasks.core.lift.mdp.commands.pose_commands import ObjectUniformPoseCommand
from isaaclab_tasks.core.lift.mdp.commands.pose_commands_cfg import ObjectUniformPoseCommandCfg

if TYPE_CHECKING:
    from isaaclab.assets import DeformableObject
    from isaaclab.envs import ManagerBasedEnv


class DeformableUniformPoseCommand(ObjectUniformPoseCommand):
    """Uniform position command for a deformable object, tracked by its center of mass.

    Deformable objects expose no root orientation, so the target is tracked with the COM
    (:attr:`~isaaclab.assets.DeformableObject.data.root_pos_w`) and only ``position_only``
    commands are supported.

    The success visualizer asset may be a static asset (``AssetBaseCfg``), which has no
    runtime view. In that case its world position is the fixed spawn offset from the
    environment origins.
    """

    cfg: DeformableUniformPoseCommandCfg
    """Configuration for the command generator."""

    object: DeformableObject
    """The deformable object tracked by the command."""

    def __init__(self, cfg: DeformableUniformPoseCommandCfg, env: ManagerBasedEnv):
        if not cfg.position_only:
            raise ValueError("DeformableUniformPoseCommand only supports position_only commands.")
        super().__init__(cfg, env)

        # static assets are stored as their config, so their world position is constant
        if isinstance(self.success_vis_asset, AssetBaseCfg):
            offset = torch.tensor(self.success_vis_asset.init_state.pos, device=self.device)
            self._static_success_vis_pos_w = env.scene.env_origins + offset
        else:
            self._static_success_vis_pos_w = None

    def _update_metrics(self):
        # transform command from base frame to simulation world frame
        self.pose_command_w[:, :3], self.pose_command_w[:, 3:] = combine_frame_transforms(
            self.robot.data.root_pos_w.torch,
            self.robot.data.root_quat_w.torch,
            self.pose_command_b[:, :3],
            self.pose_command_b[:, 3:],
        )
        com_w = self.object.data.root_pos_w.torch
        self.metrics["position_error"] = torch.linalg.norm(self.pose_command_w[:, :3] - com_w, dim=-1)

        if self.success_vis_asset is None:
            return
        # same success radius as the goal markers of the base class
        success_id = (self.metrics["position_error"] < 0.05).int()
        if self._static_success_vis_pos_w is not None:
            vis_pos_w = self._static_success_vis_pos_w
        else:
            vis_pos_w = self.success_vis_asset.data.root_pos_w.torch
        self.success_visualizer.visualize(vis_pos_w, marker_indices=success_id)


@configclass
class DeformableUniformPoseCommandCfg(ObjectUniformPoseCommandCfg):
    """Configuration for the deformable uniform pose command generator."""

    class_type: type[DeformableUniformPoseCommand] | str = "{DIR}.pose_commands:DeformableUniformPoseCommand"
