# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
from typing import Dict, Optional, Sequence

from omni.isaac.core.materials import PhysicsMaterial
from omni.isaac.core.prims import RigidPrimView
from pxr import PhysxSchema

import omni.isaac.orbit.utils.kit as kit_utils
from omni.isaac.orbit.utils.math import combine_frame_transforms, quat_rotate_inverse, subtract_frame_transforms

from ..robot_base import RobotBase
from .legged_robot_cfg import LeggedRobotCfg
from .legged_robot_data import LeggedRobotData


class LeggedRobot(RobotBase):
    """Class for handling a floating-base legged robot."""

    cfg: LeggedRobotCfg
    """Configuration for the legged robot."""
    feet_bodies: Dict[str, RigidPrimView]
    """Rigid body view for feet of the robot.

    Dictionary with keys as the foot names and values as the corresponding rigid body view
    in the robot.
    """

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

    def spawn(self, prim_path: str, translation: Sequence[float] = None, orientation: Sequence[float] = None):
        # spawn the robot and set its location
        super().spawn(prim_path, translation, orientation)
        # add physics material to the feet bodies!
        if self.cfg.physics_material is not None:
            # -- resolve material path
            material_path = self.cfg.physics_material.prim_path
            if not material_path.startswith("/"):
                material_path = prim_path + "/" + material_path
            # -- create material
            material = PhysicsMaterial(
                prim_path=material_path,
                static_friction=self.cfg.physics_material.static_friction,
                dynamic_friction=self.cfg.physics_material.dynamic_friction,
                restitution=self.cfg.physics_material.restitution,
            )
            # -- enable patch-friction: yields better results!
            physx_material_api = PhysxSchema.PhysxMaterialAPI.Apply(material.prim)
            physx_material_api.CreateImprovePatchFrictionAttr().Set(True)
            # -- bind material to feet
            for foot_cfg in self.cfg.feet_info.values():
                kit_utils.apply_nested_physics_material(f"{prim_path}/{foot_cfg.body_name}", material.prim_path)

    def initialize(self, prim_paths_expr: Optional[str] = None):
        # initialize parent handles
        super().initialize(prim_paths_expr)
        # create handles
        # -- feet
        self.feet_bodies: Dict[str, RigidPrimView] = dict()
        for foot_name, body_name in self._feet_body_name.items():
            # create rigid body view to track
            feet_body = RigidPrimView(
                prim_paths_expr=f"{self._prim_paths_expr}/{body_name}", name=foot_name, reset_xform_properties=False
            )
            feet_body.initialize()
            # add to list
            self.feet_bodies[foot_name] = feet_body

    def reset_buffers(self, env_ids: Optional[Sequence[int]] = None):
        # reset parent buffers
        super().reset_buffers(env_ids)
        # use ellipses object to skip initial indices.
        if env_ids is None:
            env_ids = ...
        # reset timers
        self._ongoing_feet_air_time[env_ids] = 0.0
        self._data.feet_air_time[env_ids] = 0.0

    def update_buffers(self, dt: float):
        # update parent buffers
        super().update_buffers(dt)
        # frame states
        # -- root frame in base
        self._data.root_vel_b[:, 0:3] = quat_rotate_inverse(self._data.root_quat_w, self._data.root_lin_vel_w)
        self._data.root_vel_b[:, 3:6] = quat_rotate_inverse(self._data.root_quat_w, self._data.root_ang_vel_w)
        self._data.projected_gravity_b[:] = quat_rotate_inverse(self._data.root_quat_w, self._GRAVITY_VEC_W)
        # -- feet
        # TODO: This can be sped up by combining all feet into one regex expression.
        for index, (foot_name, body) in enumerate(self.feet_bodies.items()):
            # extract foot offset information
            foot_pos_offset = self._feet_pos_offset[foot_name]
            foot_rot_offset = self._feet_rot_offset[foot_name]
            # world frame
            # -- foot frame in world: world -> shank frame -> foot frame
            shank_position_w, shank_quat_w = body.get_world_poses(indices=self._ALL_INDICES, clone=False)
            position_w, quat_w = combine_frame_transforms(
                shank_position_w, shank_quat_w, foot_pos_offset, foot_rot_offset
            )
            self._data.feet_state_w[:, index, 0:3] = position_w
            self._data.feet_state_w[:, index, 3:7] = quat_w
            # TODO: Transformation velocities from hand to end-effector?
            self._data.feet_state_w[:, index, 7:] = body.get_velocities(indices=self._ALL_INDICES, clone=False)
            # base frame
            position_b, quat_b = subtract_frame_transforms(
                self._data.root_state_w[:, 0:3],
                self._data.root_state_w[:, 3:7],
                self._data.feet_state_w[:, index, 0:3],
                self._data.feet_state_w[:, index, 3:7],
            )
            self._data.feet_state_b[:, index, 0:3] = position_b
            self._data.feet_state_b[:, index, 3:7] = quat_b
            self._data.feet_state_b[:, index, 7:10] = quat_rotate_inverse(
                self._data.root_quat_w, self._data.feet_state_w[:, index, 7:10]
            )
            self._data.feet_state_b[:, index, 10:13] = quat_rotate_inverse(
                self._data.root_quat_w, self._data.feet_state_w[:, index, 10:13]
            )
        # TODO: contact forces -- Waiting for contact sensors in IsaacSim.
        #   For now, use heuristics for flat terrain to say feet are in contact.
        # air times
        # -- update ongoing timer for feet air
        self._ongoing_feet_air_time += dt
        # -- check contact state of feet
        is_feet_contact = self._data.feet_state_w[:, :, 2] < 0.03
        is_feet_first_contact = (self._ongoing_feet_air_time > 0) * is_feet_contact
        # -- update buffers
        self._data.feet_air_time = self._ongoing_feet_air_time * is_feet_first_contact
        # -- reset timers for feet that are in contact for the first time
        self._ongoing_feet_air_time *= ~is_feet_contact

    """
    Internal helpers - Override.
    """

    def _process_info_cfg(self) -> None:
        """Post processing of configuration parameters."""
        # process parent config
        super()._process_info_cfg()
        # feet offsets
        self._feet_body_name: Dict[str, str] = dict.fromkeys(self.cfg.feet_info)
        self._feet_pos_offset: Dict[str, torch.Tensor] = dict.fromkeys(self.cfg.feet_info)
        self._feet_rot_offset: Dict[str, torch.Tensor] = dict.fromkeys(self.cfg.feet_info)
        for foot_name, foot_cfg in self.cfg.feet_info.items():
            # -- body name
            self._feet_body_name[foot_name] = foot_cfg.body_name
            # -- position
            foot_pos_offset = torch.tensor(foot_cfg.pos_offset, device=self.device)
            self._feet_pos_offset[foot_name] = foot_pos_offset.repeat(self.count, 1)
            # -- orientation
            foot_rot_offset = torch.tensor(foot_cfg.rot_offset, device=self.device)
            self._feet_rot_offset[foot_name] = foot_rot_offset.repeat(self.count, 1)

    def _create_buffers(self):
        """Create buffers for storing data."""
        # process parent buffers
        super()._create_buffers()
        # history
        self._ongoing_feet_air_time = torch.zeros(self.count, len(self._feet_body_name), device=self.device)
        # constants
        # TODO: get gravity direction from stage.
        self._GRAVITY_VEC_W = torch.tensor((0.0, 0.0, -1.0), device=self.device).repeat(self.count, 1)
        self._FORWARD_VEC_B = torch.tensor((1.0, 0.0, 0.0), device=self.device).repeat(self.count, 1)

        # frame states
        # -- base
        self._data.root_vel_b = torch.zeros(self.count, 6, dtype=torch.float, device=self.device)
        self._data.projected_gravity_b = torch.zeros(self.count, 3, dtype=torch.float, device=self.device)
        # -- feet
        self._data.feet_state_w = torch.zeros(self.count, len(self._feet_body_name), 13, device=self.device)
        self._data.feet_state_b = torch.zeros_like(self._data.feet_state_w)
        # -- air time
        self._data.feet_air_time = torch.zeros(self.count, len(self._feet_body_name), device=self.device)
