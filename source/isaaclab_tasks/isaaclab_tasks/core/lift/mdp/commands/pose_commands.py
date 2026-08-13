# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Sub-module containing command generators for pose tracking."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import AssetBaseCfg
from isaaclab.managers import CommandTerm
from isaaclab.utils.leapp import POSE7_ELEMENT_NAMES
from isaaclab.utils.math import combine_frame_transforms, compute_pose_error, quat_from_euler_xyz, quat_unique

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, CableObject, DeformableObject, RigidObject
    from isaaclab.envs import ManagerBasedEnv

    from .pose_commands_cfg import (
        CableUniformPoseCommandCfg,
        DeformableUniformPoseCommandCfg,
        ObjectUniformPoseCommandCfg,
    )


class ObjectUniformPoseCommand(CommandTerm):
    """Uniform pose command generator for an object (in the robot base frame).

    This command term samples target object poses by:
      • Drawing (x, y, z) uniformly within configured Cartesian bounds, and
      • Drawing roll-pitch-yaw uniformly within configured ranges, then converting
        to a quaternion (x, y, z, w). Optionally makes quaternions unique by enforcing
        a positive real part.

    Frames:
        Targets are defined in the robot's *base frame*. For metrics/visualization,
        targets are transformed into the *world frame* using the robot root pose.

    Outputs:
        The command buffer has shape (num_envs, 7): ``(x, y, z, qx, qy, qz, qw)``.

    Metrics:
        `position_error` and `orientation_error` are computed between the commanded
        world-frame pose and the object's current world-frame pose.

    Config:
        `cfg` must provide the sampling ranges, whether to enforce quaternion uniqueness,
        and optional visualization settings.
    """

    cfg: ObjectUniformPoseCommandCfg
    """Configuration for the command generator."""

    def __init__(self, cfg: ObjectUniformPoseCommandCfg, env: ManagerBasedEnv):
        """Initialize the command generator class.

        Args:
            cfg: The configuration parameters for the command generator.
            env: The environment object.
        """
        # initialize the base class
        super().__init__(cfg, env)

        # extract the robot and body index for which the command is generated
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.object: RigidObject = env.scene[cfg.object_name]
        self.success_vis_asset: RigidObject | AssetBaseCfg | None
        if cfg.success_vis_asset_name in env.scene.keys():
            self.success_vis_asset = env.scene[cfg.success_vis_asset_name]
        else:
            self.success_vis_asset = None
        if isinstance(self.success_vis_asset, AssetBaseCfg):
            offset = torch.tensor(self.success_vis_asset.init_state.pos, device=self.device)
            self._static_success_vis_pos_w = env.scene.env_origins + offset
        else:
            self._static_success_vis_pos_w = None

        # create buffers
        # -- commands: (x, y, z, qx, qy, qz, qw) in root frame
        self.pose_command_b = torch.zeros(self.num_envs, 7, device=self.device)
        self.pose_command_b[:, 3] = 1.0
        self.pose_command_w = torch.zeros_like(self.pose_command_b)
        # -- metrics
        self.metrics["position_error"] = torch.zeros(self.num_envs, device=self.device)
        if not self.cfg.position_only:
            self.metrics["orientation_error"] = torch.zeros(self.num_envs, device=self.device)
        from isaaclab.markers import VisualizationMarkers

        self.success_visualizer = VisualizationMarkers(self.cfg.success_visualizer_cfg)
        self.success_visualizer.set_visibility(True)
        if self.success_vis_asset is not None:
            self.success_visualizer.visualize(self._get_success_vis_pos_w())

        # adds (optional) cmd kind and element names for leapp export
        # during export, semantic data about this command will be used to annotate the command input
        self.cfg.cmd_kind = self.cfg.cmd_kind or "command/body/pose"
        self.cfg.element_names = self.cfg.element_names or POSE7_ELEMENT_NAMES

    def __str__(self) -> str:
        msg = "ObjectUniformPoseCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The desired pose command. Shape is (num_envs, 7).

        The first three elements correspond to the position, followed by the quaternion orientation in (x, y, z, w).
        """
        return self.pose_command_b

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        # transform command from base frame to simulation world frame
        self.pose_command_w[:, :3], self.pose_command_w[:, 3:] = combine_frame_transforms(
            self.robot.data.root_pos_w.torch,
            self.robot.data.root_quat_w.torch,
            self.pose_command_b[:, :3],
            self.pose_command_b[:, 3:],
        )
        # compute the error
        object_root_pose_w = self.object.data.root_link_pose_w.torch
        pos_error, rot_error = compute_pose_error(
            self.pose_command_w[:, :3],
            self.pose_command_w[:, 3:],
            object_root_pose_w[:, :3],
            object_root_pose_w[:, 3:7],
        )
        self.metrics["position_error"] = torch.linalg.norm(pos_error, dim=-1)

        success_id = self.metrics["position_error"] < 0.05
        if not self.cfg.position_only:
            self.metrics["orientation_error"] = torch.linalg.norm(rot_error, dim=-1)
            success_id &= self.metrics["orientation_error"] < 0.5
        if self.success_vis_asset is not None:
            self.success_visualizer.visualize(self._get_success_vis_pos_w(), marker_indices=success_id.int())

    def _get_success_vis_pos_w(self) -> torch.Tensor:
        """Return the success visualization positions in the world frame."""
        if self._static_success_vis_pos_w is not None:
            return self._static_success_vis_pos_w
        return self.success_vis_asset.data.root_pos_w.torch

    def _resample_command(self, env_ids: Sequence[int]):
        # sample new pose targets
        # -- position
        r = torch.empty(len(env_ids), device=self.device)
        self.pose_command_b[env_ids, 0] = r.uniform_(*self.cfg.ranges.pos_x)
        self.pose_command_b[env_ids, 1] = r.uniform_(*self.cfg.ranges.pos_y)
        self.pose_command_b[env_ids, 2] = r.uniform_(*self.cfg.ranges.pos_z)
        # -- orientation
        euler_angles = torch.zeros_like(self.pose_command_b[env_ids, :3])
        euler_angles[:, 0].uniform_(*self.cfg.ranges.roll)
        euler_angles[:, 1].uniform_(*self.cfg.ranges.pitch)
        euler_angles[:, 2].uniform_(*self.cfg.ranges.yaw)
        quat = quat_from_euler_xyz(euler_angles[:, 0], euler_angles[:, 1], euler_angles[:, 2])
        # make sure the quaternion has real part as positive
        self.pose_command_b[env_ids, 3:] = quat_unique(quat) if self.cfg.make_quat_unique else quat

    def _update_command(self):
        pass

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "goal_visualizer"):
                from isaaclab.markers import VisualizationMarkers

                self.goal_visualizer = VisualizationMarkers(self.cfg.goal_pose_visualizer_cfg)
                self.curr_visualizer = VisualizationMarkers(self.cfg.curr_pose_visualizer_cfg)
            # set their visibility to true
            self.goal_visualizer.set_visibility(True)
            self.curr_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_visualizer"):
                self.goal_visualizer.set_visibility(False)
                self.curr_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # check if robot is initialized
        # note: this is needed in-case the robot is de-initialized. we can't access the data
        if not self.robot.is_initialized:
            return
        # update the markers
        if not self.cfg.position_only:
            # -- goal pose
            self.goal_visualizer.visualize(self.pose_command_w[:, :3], self.pose_command_w[:, 3:])
            # -- current object pose
            obj_pos = self.object.data.root_pos_w.torch
            obj_quat = self.object.data.root_quat_w.torch
            self.curr_visualizer.visualize(obj_pos, obj_quat)
        else:
            obj_pos = self.object.data.root_pos_w.torch
            distance = torch.linalg.norm(self.pose_command_w[:, :3] - obj_pos, dim=1)
            success_id = (distance < 0.05).int()
            # note: since marker indices for position is 1(far) and 2(near), we can simply shift the success_id by 1.
            # -- goal position
            self.goal_visualizer.visualize(self.pose_command_w[:, :3], marker_indices=success_id + 1)
            # -- current object position
            self.curr_visualizer.visualize(obj_pos, marker_indices=success_id + 1)


class DeformableUniformPoseCommand(ObjectUniformPoseCommand):
    """Uniform position command for a deformable object, tracked by its center of mass.

    Deformable objects expose no root orientation, so the target is tracked with the COM
    (:attr:`~isaaclab.assets.DeformableObject.data.root_pos_w`) and only ``position_only``
    commands are supported.
    """

    cfg: DeformableUniformPoseCommandCfg
    """Configuration for the command generator."""

    object: DeformableObject
    """The deformable object tracked by the command."""

    def __init__(self, cfg: DeformableUniformPoseCommandCfg, env: ManagerBasedEnv):
        if not cfg.position_only:
            raise ValueError("DeformableUniformPoseCommand only supports position_only commands.")
        super().__init__(cfg, env)

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
        self.success_visualizer.visualize(self._get_success_vis_pos_w(), marker_indices=success_id)


class CableUniformPoseCommand(ObjectUniformPoseCommand):
    """Uniform position command tracked by one cable segment."""

    cfg: CableUniformPoseCommandCfg
    """Configuration for the command generator."""

    object: CableObject
    """Cable tracked by the command."""

    def __init__(self, cfg: CableUniformPoseCommandCfg, env: ManagerBasedEnv):
        if not cfg.position_only:
            raise ValueError("CableUniformPoseCommand only supports position_only commands.")
        super().__init__(cfg, env)
        if not 0 <= cfg.segment_index < self.object.num_segments:
            raise ValueError(f"segment_index must be in [0, {self.object.num_segments}), received {cfg.segment_index}.")

    def _segment_position_w(self) -> torch.Tensor:
        return self.object.data.segment_pose_w.torch[:, self.cfg.segment_index, :3]

    def _update_metrics(self):
        self.pose_command_w[:, :3], self.pose_command_w[:, 3:] = combine_frame_transforms(
            self.robot.data.root_pos_w.torch,
            self.robot.data.root_quat_w.torch,
            self.pose_command_b[:, :3],
            self.pose_command_b[:, 3:],
        )
        segment_pos_w = self._segment_position_w()
        self.metrics["position_error"] = torch.linalg.norm(self.pose_command_w[:, :3] - segment_pos_w, dim=-1)

        if self.success_vis_asset is None:
            return
        success_id = (self.metrics["position_error"] < 0.05).int()
        self.success_visualizer.visualize(self._get_success_vis_pos_w(), marker_indices=success_id)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return
        segment_pos_w = self._segment_position_w()
        distance = torch.linalg.norm(self.pose_command_w[:, :3] - segment_pos_w, dim=1)
        marker_indices = (distance < 0.05).int() + 1
        self.goal_visualizer.visualize(self.pose_command_w[:, :3], marker_indices=marker_indices)
        self.curr_visualizer.visualize(segment_pos_w, marker_indices=marker_indices)
