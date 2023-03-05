# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import re
import torch
from typing import Dict, Optional, Sequence

from omni.isaac.core.materials import PhysicsMaterial
from omni.isaac.core.prims import RigidPrimView
from pxr import PhysxSchema

import omni.isaac.orbit.utils.kit as kit_utils
from omni.isaac.orbit.utils.math import combine_frame_transforms, quat_rotate_inverse, subtract_frame_transforms

from ..robot_base import RobotBase
from .single_arm_cfg import SingleArmManipulatorCfg
from .single_arm_data import SingleArmManipulatorData


class SingleArmManipulator(RobotBase):
    """Class for handling a fixed-base robot arm with a tool on it."""

    cfg: SingleArmManipulatorCfg
    """Configuration for the robot."""
    ee_parent_body: RigidPrimView
    """Rigid body view for the end-effector parent body."""
    tool_site_bodies: Dict[str, RigidPrimView]
    """Rigid body views for the tool sites.

    Dictionary with keys as the site names and values as the corresponding rigid body view
    in the articulated object.
    """

    def __init__(self, cfg: SingleArmManipulatorCfg):
        """Initialize the robot class.

        Args:
            cfg (SingleArmManipulatorCfg): The configuration instance.
        """
        # initialize parent
        super().__init__(cfg)
        # container for data access
        self._data = SingleArmManipulatorData()

    """
    Properties
    """

    @property
    def arm_num_dof(self) -> int:
        """Number of DOFs in the robot arm."""
        return self.cfg.meta_info.arm_num_dof

    @property
    def tool_num_dof(self) -> int:
        """Number of DOFs in the robot tool/gripper."""
        return self.cfg.meta_info.tool_num_dof

    @property
    def data(self) -> SingleArmManipulatorData:
        """Data related to articulation."""
        return self._data

    """
    Operations.
    """

    def spawn(self, prim_path: str, translation: Sequence[float] = None, orientation: Sequence[float] = None):
        # spawn the robot and set its location
        super().spawn(prim_path, translation, orientation)
        # alter physics collision properties
        kit_utils.set_nested_collision_properties(prim_path, contact_offset=0.02, rest_offset=0.0)
        # add physics material to the tool sites bodies!
        if self.cfg.physics_material is not None and self.cfg.meta_info.tool_sites_names is not None:
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
            for site_name in self.cfg.meta_info.tool_sites_names:
                kit_utils.apply_nested_physics_material(f"{prim_path}/{site_name}", material.prim_path)

    def initialize(self, prim_paths_expr: Optional[str] = None):
        # initialize parent handles
        super().initialize(prim_paths_expr)
        # create handles
        # -- ee frame
        self.ee_parent_body = RigidPrimView(
            prim_paths_expr=f"{self._prim_paths_expr}/{self.cfg.ee_info.body_name}", reset_xform_properties=False
        )
        self.ee_parent_body.initialize()
        # -- gripper
        if self.tool_sites_indices is not None:
            self.tool_site_bodies: Dict[str, RigidPrimView] = dict()
            for name in self.tool_sites_indices:
                # create rigid body view to track
                site_body = RigidPrimView(
                    prim_paths_expr=f"{self._prim_paths_expr}/{name}", reset_xform_properties=False
                )
                site_body.initialize()
                # add to list
                self.tool_site_bodies[name] = site_body
        else:
            self.tool_site_bodies = None

    def update_buffers(self, dt: float):
        # update parent buffers
        super().update_buffers(dt)
        # frame states
        # -- ee frame in world: world -> hand frame -> ee frame
        hand_position_w, hand_quat_w = self.ee_parent_body.get_world_poses(indices=self._ALL_INDICES, clone=False)
        position_w, quat_w = combine_frame_transforms(
            hand_position_w, hand_quat_w, self._ee_pos_offset, self._ee_rot_offset
        )
        self._data.ee_state_w[:, 0:3] = position_w
        self._data.ee_state_w[:, 3:7] = quat_w
        # TODO: Transformation velocities from hand to end-effector?
        self._data.ee_state_w[:, 7:] = self.ee_parent_body.get_velocities(indices=self._ALL_INDICES, clone=False)
        # ee frame in body
        position_b, quat_b = subtract_frame_transforms(
            self._data.root_state_w[:, 0:3],
            self._data.root_state_w[:, 3:7],
            self._data.ee_state_w[:, 0:3],
            self._data.ee_state_w[:, 3:7],
        )
        self._data.ee_state_b[:, 0:3] = position_b
        self._data.ee_state_b[:, 3:7] = quat_b
        self._data.ee_state_b[:, 7:10] = quat_rotate_inverse(self._data.root_quat_w, self._data.ee_state_w[:, 7:10])
        self._data.ee_state_b[:, 10:13] = quat_rotate_inverse(self._data.root_quat_w, self._data.ee_state_w[:, 10:13])
        # -- tool sites
        # TODO: This can be sped up by combining all tool sites into one regex expression.
        if self.tool_site_bodies is not None:
            for index, body in enumerate(self.tool_site_bodies.values()):
                # world frame
                position_w, quat_w = body.get_world_poses(indices=self._ALL_INDICES, clone=False)
                self._data.tool_sites_state_w[:, index, 0:3] = position_w
                self._data.tool_sites_state_w[:, index, 3:7] = quat_w
                self._data.tool_sites_state_w[:, index, 7:] = body.get_velocities(
                    indices=self._ALL_INDICES, clone=False
                )
                # base frame
                position_b, quat_b = subtract_frame_transforms(
                    self._data.root_state_w[:, 0:3],
                    self._data.root_state_w[:, 3:7],
                    self._data.tool_sites_state_w[:, index, 0:3],
                    self._data.tool_sites_state_w[:, index, 3:7],
                )
                self._data.tool_sites_state_b[:, index, 0:3] = position_b
                self._data.tool_sites_state_b[:, index, 3:7] = quat_b
                self._data.tool_sites_state_b[:, index, 7:10] = quat_rotate_inverse(
                    self._data.root_quat_w, self._data.tool_sites_state_w[:, index, 7:10]
                )
                self._data.tool_sites_state_b[:, index, 10:13] = quat_rotate_inverse(
                    self._data.root_quat_w, self._data.tool_sites_state_w[:, index, 10:13]
                )
        # update optional data
        self._update_optional_buffers()

    """
    Internal helpers - Override.
    """

    def _process_info_cfg(self) -> None:
        """Post processing of configuration parameters."""
        # process parent config
        super()._process_info_cfg()
        # resolve regex expressions for indices
        # -- end-effector body
        self.ee_body_index = -1
        for body_index, body_name in enumerate(self.body_names):
            if re.fullmatch(self.cfg.ee_info.body_name, body_name):
                self.ee_body_index = body_index
        if self.ee_body_index == -1:
            raise ValueError(f"Could not find end-effector body with name: {self.cfg.ee_info.body_name}")
        # -- tool sites
        if self.cfg.meta_info.tool_sites_names:
            tool_sites_names = list()
            tool_sites_indices = list()
            for body_index, body_name in enumerate(self.body_names):
                for re_key in self.cfg.meta_info.tool_sites_names:
                    if re.fullmatch(re_key, body_name):
                        tool_sites_names.append(body_name)
                        tool_sites_indices.append(body_index)
            # check valid indices
            if len(tool_sites_names) == 0:
                raise ValueError(f"Could not find any tool sites with names: {self.cfg.meta_info.tool_sites_names}")
            # create dictionary to map names to indices
            self.tool_sites_indices: Dict[str, int] = dict(zip(tool_sites_names, tool_sites_indices))
        else:
            self.tool_sites_indices = None
        # end-effector offsets
        # -- position
        ee_pos_offset = torch.tensor(self.cfg.ee_info.pos_offset, device=self.device)
        self._ee_pos_offset = ee_pos_offset.repeat(self.count, 1)
        # -- orientation
        ee_rot_offset = torch.tensor(self.cfg.ee_info.rot_offset, device=self.device)
        self._ee_rot_offset = ee_rot_offset.repeat(self.count, 1)

    def _create_buffers(self):
        """Create buffers for storing data."""
        # process parent buffers
        super()._create_buffers()

        # -- frame states
        self._data.ee_state_w = torch.zeros_like(self._data.root_state_w)
        self._data.ee_state_b = torch.zeros_like(self._data.root_state_w)
        if self.tool_sites_indices is not None:
            self._data.tool_sites_state_w = torch.zeros(
                self.count, len(self.tool_sites_indices), 13, device=self.device
            )
            self._data.tool_sites_state_b = torch.zeros_like(self._data.tool_sites_state_w)
        # -- dof states
        self._data.arm_dof_pos = self._data.dof_pos[:, : self.arm_num_dof]
        self._data.arm_dof_vel = self._data.dof_vel[:, : self.arm_num_dof]
        self._data.arm_dof_acc = self._data.dof_acc[:, : self.arm_num_dof]
        self._data.tool_dof_pos = self._data.dof_pos[:, self.arm_num_dof :]
        self._data.tool_dof_vel = self._data.dof_vel[:, self.arm_num_dof :]
        self._data.tool_dof_acc = self._data.dof_acc[:, self.arm_num_dof :]
        # -- dynamic states
        if self.cfg.data_info.enable_jacobian:
            self._data.ee_jacobian = torch.zeros(self.count, 6, self.arm_num_dof, device=self.device)
        if self.cfg.data_info.enable_mass_matrix:
            self._data.mass_matrix = torch.zeros(self.count, self.arm_num_dof, self.arm_num_dof, device=self.device)
        if self.cfg.data_info.enable_coriolis:
            self._data.coriolis = torch.zeros(self.count, self.arm_num_dof, device=self.device)
        if self.cfg.data_info.enable_gravity:
            self._data.gravity = torch.zeros(self.count, self.arm_num_dof, device=self.device)

    def _update_optional_buffers(self):
        """Update buffers from articulation that are optional."""
        # Note: we implement this function here to allow inherited classes decide whether these
        #   quantities need to be updated similarly or not.
        # -- dynamic state (note: tools don't contribute towards these quantities)
        # jacobian
        if self.cfg.data_info.enable_jacobian:
            jacobians = self.articulations.get_jacobians(indices=self._ALL_INDICES, clone=False)
            # Returned jacobian: [batch, body, 6, dof] does not include the base body (i.e. the first link).
            # So we need to subtract 1 from the body index to get the correct jacobian.
            self._data.ee_jacobian[:] = jacobians[:, self.ee_body_index - 1, :, : self.arm_num_dof]
        # mass matrix
        if self.cfg.data_info.enable_mass_matrix:
            mass_matrices = self.articulations.get_mass_matrices(indices=self._ALL_INDICES, clone=False)
            self._data.mass_matrix[:] = mass_matrices[:, : self.arm_num_dof, : self.arm_num_dof]
        # coriolis
        if self.cfg.data_info.enable_coriolis:
            forces = self.articulations.get_coriolis_and_centrifugal_forces(indices=self._ALL_INDICES, clone=False)
            self._data.coriolis[:] = forces[:, : self.arm_num_dof]
        # gravity
        if self.cfg.data_info.enable_gravity:
            gravity = self.articulations.get_generalized_gravity_forces(indices=self._ALL_INDICES, clone=False)
            self._data.gravity[:] = gravity[:, : self.arm_num_dof]
