# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
from typing import Optional, Sequence

from omni.isaac.orbit.utils.math import combine_frame_transforms, quat_rotate_inverse, subtract_frame_transforms

from ..robot_base import RobotBase
from .legged_robot_cfg import LeggedRobotCfg
from .legged_robot_data import LeggedRobotData


class LeggedRobot(RobotBase):
    """Class for handling a floating-base legged robot."""

    cfg: LeggedRobotCfg
    """Configuration for the legged robot."""

    def __init__(self, cfg: LeggedRobotCfg):
        """Initialize the robot class.

        Args:
            cfg (LeggedRobotCfg): The configuration instance.
        """
        # initialize parent
        super().__init__(cfg)
        # container for data access
        self._data = LeggedRobotData()

    """
    Properties
    """

    @property
    def data(self) -> LeggedRobotData:
        """Data related to articulation."""
        return self._data

    @property
    def feet_names(self) -> Sequence[str]:
        """Names of the feet."""
        return list(self.cfg.feet_info.keys())

    """
    Operations.
    """

    def initialize(self, prim_paths_expr: Optional[str] = None):
        # default prim path if not cloned
        super().initialize(prim_paths_expr)
        if prim_paths_expr is None:
            if self._is_spawned is None:
                raise RuntimeError("Failed to initialize robot. Please provide a valid 'prim_paths_expr'.")
            # -- use spawn path
            self._prim_paths_expr = self._spawn_prim_path
        else:
            self._prim_paths_expr = prim_paths_expr
        # combine body names for feet sensors into single regex
        self.feet_indices, _ = self.find_bodies(self.feet_names)
        # initialize parent handles

    def reset_buffers(self, env_ids: Optional[Sequence[int]] = None):
        # reset parent buffers
        super().reset_buffers(env_ids)
        # use ellipses object to skip initial indices.
        if env_ids is None:
            env_ids = ...

    def update_buffers(self, dt: float):
        # update parent buffers
        super().update_buffers(dt)
        # frame states
        # -- root frame in base
        self._data.root_vel_b[:, 0:3] = quat_rotate_inverse(self._data.root_quat_w, self._data.root_lin_vel_w)
        self._data.root_vel_b[:, 3:6] = quat_rotate_inverse(self._data.root_quat_w, self._data.root_ang_vel_w)
        self._data.projected_gravity_b[:] = quat_rotate_inverse(self._data.root_quat_w, self._GRAVITY_VEC_W)
        # -- feet
        # world frame
        # -- foot frame in world: world -> shank frame -> foot frame
        position_w, quat_w = self._body_view.get_world_poses(clone=False)
        position_w = position_w.view(-1, self.num_bodies, 3)[:, self.feet_indices].flatten(0, 1)
        quat_w = quat_w.view(-1, self.num_bodies, 4)[:, self.feet_indices].flatten(0, 1)
        position_w, quat_w = combine_frame_transforms(position_w, quat_w, self._feet_pos_offset, self._feet_rot_offset)
        self._data.feet_state_w[:, :, 0:3] = position_w.view(self.count, -1, 3)
        self._data.feet_state_w[:, :, 3:7] = quat_w.view(self.count, -1, 4)
        self._data.feet_state_w[:, :, 7:] = self._body_view.get_velocities(clone=False).view(-1, self.num_bodies, 6)[
            :, self.feet_indices
        ]  # FIXME this is not correct
        # base frame
        num_feet = len(self.feet_names)
        # -- cast data to match shape
        root_pose_w = self._data.root_state_w[:, 0:7].unsqueeze(1).expand(self.count, num_feet, 7)
        position_b, quat_b = subtract_frame_transforms(
            root_pose_w[:, :, 0:3],
            root_pose_w[:, :, 3:7],
            self._data.feet_state_w[:, :, 0:3],
            self._data.feet_state_w[:, :, 3:7],
        )
        self._data.feet_pose_b[:, :, 0:3] = position_b
        self._data.feet_pose_b[:, :, 3:7] = quat_b

    """
    Internal helpers - Override.
    """

    def _process_info_cfg(self) -> None:
        """Post processing of configuration parameters."""
        # process parent config
        super()._process_info_cfg()
        # check that order of feet is consistent
        # feet offsets
        feet_pos_offset = dict()
        feet_rot_offset = dict()
        # Note: we need to make sure reading config is consistent with the order of bodies in simulation
        # iterate over feet bodies config
        for foot_name in self.feet_names:
            # find the foot config that matches the body name
            for foot_cfg_name, foot_cfg in self.cfg.feet_info.items():
                if foot_cfg_name == foot_name:
                    feet_pos_offset[foot_name] = torch.tensor(foot_cfg.pos_offset, device=self.device)
                    feet_rot_offset[foot_name] = torch.tensor(foot_cfg.rot_offset, device=self.device)
        # concatenate into single tensors: (num_feet, K) -> (count * num_feet, K)
        self._feet_pos_offset = torch.stack(list(feet_pos_offset.values()), dim=0).repeat(self.count, 1)
        self._feet_rot_offset = torch.stack(list(feet_rot_offset.values()), dim=0).repeat(self.count, 1)
        # store feet names

    def _create_buffers(self):
        """Create buffers for storing data."""
        # process parent buffers
        super()._create_buffers()
        # constants
        self._GRAVITY_VEC_W = torch.tensor((0.0, 0.0, -1.0), device=self.device).repeat(self.count, 1)
        self._FORWARD_VEC_B = torch.tensor((1.0, 0.0, 0.0), device=self.device).repeat(self.count, 1)

        # frame states
        # -- base
        self._data.root_vel_b = torch.zeros(self.count, 6, dtype=torch.float, device=self.device)
        self._data.projected_gravity_b = torch.zeros(self.count, 3, dtype=torch.float, device=self.device)
        # -- feet
        num_feet = len(self.feet_names)
        self._data.feet_state_w = torch.zeros(self.count, num_feet, 13, device=self.device)
        self._data.feet_pose_b = torch.zeros(self.count, num_feet, 7, device=self.device)
