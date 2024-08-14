# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import carb
import omni.isaac.core.utils.prims as prim_utils
import omni.physics.tensors.impl.api as physx

import omni.isaac.lab.sim as sim_utils

from ..asset_base import AssetBase
from .deformable_object_data import DeformableObjectData

if TYPE_CHECKING:
    from .deformable_object_cfg import DeformableObjectCfg


class DeformableObject(AssetBase):
    """Class for handling deformable objects."""

    cfg: DeformableObjectCfg
    """Configuration instance for the deformable object."""

    def __init__(self, cfg: DeformableObjectCfg):
        """Initialize the deformable object.

        Args:
            cfg: A configuration instance.
        """
        super().__init__(cfg)

    """
    Properties
    """

    @property
    def data(self) -> DeformableObjectData:
        return self._data

    @property
    def num_instances(self) -> int:
        return self.root_physx_view.count

    @property
    def num_bodies(self) -> int:
        """Number of bodies in the asset."""
        return 1

    @property
    def root_physx_view(self) -> physx.SoftBodyView:
        """Deformable body view for the asset (PhysX).

        Note:
            Use this view with caution. It requires handling of tensors in a specific way.
        """
        return self._root_physx_view

    @property
    def max_simulation_mesh_elements_per_body(self) -> int:
        """The maximum number of simulation mesh elements per deformable body."""
        return self.root_physx_view.max_sim_elements_per_body

    @property
    def max_simulation_mesh_vertices_per_body(self) -> int:
        """The maximum number of simulation mesh vertices per deformable body."""
        return self.root_physx_view.max_sim_vertices_per_body

    @property
    def max_collision_mesh_elements_per_body(self) -> int:
        """The maximum number of collision mesh elements per deformable body."""
        return self.root_physx_view.max_elements_per_body

    @property
    def max_collision_mesh_vertices_per_body(self) -> int:
        """The maximum number of collision mesh vertices per deformable body."""
        return self.root_physx_view.max_vertices_per_body

    """
    Operations.
    """

    def reset(self, env_ids: Sequence[int] | None = None):
        # resolve all indices
        if env_ids is None:
            env_ids = slice(None)

    def write_data_to_sim(self):
        pass

    def update(self, dt: float):
        self._data.update(dt)

    """
    Operations - Write to simulation.
    """

    def write_root_state_to_sim(self, root_state: torch.Tensor, env_ids: Sequence[int] | None = None):
        """Set the root state over selected environment indices into the simulation.

        The root state comprises of the nodal positions and velocities. All the quantities are in the simulation frame.

        Args:
            root_state: Root state in simulation frame.
                Shape is (len(env_ids), 2 * max_simulation_mesh_vertices_per_body, 3).
            env_ids: Environment indices. If :obj:`None`, then all indices are used.
        """
        # set into simulation
        self.write_root_pos_to_sim(root_state[:, : self.max_simulation_mesh_vertices_per_body, :], env_ids=env_ids)
        self.write_root_velocity_to_sim(root_state[:, self.max_simulation_mesh_vertices_per_body :, :], env_ids=env_ids)

    def write_root_pos_to_sim(self, root_pos: torch.Tensor, env_ids: Sequence[int] | None = None):
        """Set the root pos over selected environment indices into the simulation.

        The root position comprises of individual nodal positions of the simulation mesh for the deformable body.
        The positions are in the simulation frame.

        Args:
            root_pos: Nodal positions in simulation frame. Shape is (len(env_ids), max_simulation_mesh_vertices_per_body, 3).
            env_ids: Environment indices. If :obj:`None`, then all indices are used.
        """
        # resolve all indices
        physx_env_ids = env_ids
        if env_ids is None:
            env_ids = slice(None)
            physx_env_ids = self._ALL_INDICES
        # note: we need to do this here since tensors are not set into simulation until step.
        # set into internal buffers
        self._data.nodal_pos_w[env_ids] = root_pos.clone()
        # set into simulation
        self.root_physx_view.set_sim_nodal_positions(self._data.nodal_pos_w, indices=physx_env_ids)

    def write_root_velocity_to_sim(self, root_velocity: torch.Tensor, env_ids: Sequence[int] | None = None):
        """Set the root velocity over selected environment indices into the simulation.

        The root velocity comprises of individual nodal velocities of the simulation mesh for the deformable body.

        Args:
            root_velocity: Root velocities in simulation frame. Shape is (len(env_ids), max_simulation_mesh_vertices_per_body, 3).
            env_ids: Environment indices. If :obj:`None`, then all indices are used.
        """
        # resolve all indices
        physx_env_ids = env_ids
        if env_ids is None:
            env_ids = slice(None)
            physx_env_ids = self._ALL_INDICES
        # note: we need to do this here since tensors are not set into simulation until step.
        # set into internal buffers
        self._data.nodal_vel_w[env_ids] = root_velocity.clone()
        # set into simulation
        self.root_physx_view.set_sim_nodal_velocities(self._data.nodal_vel_w, indices=physx_env_ids)

    def write_simulation_mesh_kinematic_targets(self, targets: torch.Tensor, env_ids: Sequence[int] | None = None):
        """Set the kinematic targets of the simulation mesh for the deformable bodies indicated by the indices.

        The kinematic targets comprise of individual nodal positions of the simulation mesh for the deformable body
        and a flag indicating whether the node is kinematically driven or not. The positions are in the simulation frame.

        Args:
            targets: The kinematic targets comprising of nodal positions and flags. Shape is (len(env_ids), max_simulation_mesh_vertices_per_body, 4).
            env_ids: Environment indices. If :obj:`None`, then all indices are used.
        """
        # resolve all indices
        physx_env_ids = env_ids
        if env_ids is None:
            env_ids = slice(None)
            physx_env_ids = self._ALL_INDICES
        # set into simulation
        self.root_physx_view.set_sim_kinematic_targets(targets, indices=physx_env_ids)

    """
    Internal helper.
    """

    def _initialize_impl(self):
        # create simulation view
        self._physics_sim_view = physx.create_simulation_view(self._backend)
        self._physics_sim_view.set_subspace_roots("/")
        # obtain the first prim in the regex expression (all others are assumed to be a copy of this)
        template_prim = sim_utils.find_first_matching_prim(self.cfg.prim_path)
        if template_prim is None:
            raise RuntimeError(f"Failed to find prim for expression: '{self.cfg.prim_path}'.")
        template_prim_path = template_prim.GetPath().pathString
        # find first mesh path
        mesh_path = prim_utils.get_prim_path(
            prim_utils.get_first_matching_child_prim(
                template_prim_path, lambda p: prim_utils.get_prim_type_name(p) == "Mesh"
            )
        )
        # resolve mesh path back into regex expression
        mesh_path_expr = self.cfg.prim_path + mesh_path[len(template_prim_path) :]
        # -- object views
        self._root_physx_view = self._physics_sim_view.create_soft_body_view(mesh_path_expr.replace(".*", "*"))

        # log information about the deformable body
        carb.log_info(f"Deformable body initialized at: {self.cfg.prim_path}")
        carb.log_info(f"Number of instances: {self.num_instances}")
        carb.log_info(f"Number of bodies: {self.num_bodies}")

        # container for data access
        self._data = DeformableObjectData(self.root_physx_view, self.device)

        # create buffers
        self._create_buffers()
        # process configuration
        self._process_cfg()

    def _create_buffers(self):
        """Create buffers for storing data."""
        # constants
        self._ALL_INDICES = torch.arange(self.num_instances, dtype=torch.long, device=self.device)

    def _process_cfg(self):
        """Post processing of configuration parameters."""
        # default state
        # we use the initial nodal positions at spawn time as the default state
        # note: these are all in the simulation frame
        nodal_positions = self.root_physx_view.get_sim_nodal_positions()
        nodal_velocities = torch.zeros_like(nodal_positions)
        self._data.default_nodal_state_w = torch.cat((nodal_positions, nodal_velocities), dim=1)

    """
    Internal simulation callbacks.
    """

    def _invalidate_initialize_callback(self, event):
        """Invalidates the scene elements."""
        # call parent
        super()._invalidate_initialize_callback(event)
        # set all existing views to None to invalidate them
        self._physics_sim_view = None
        self._root_physx_view = None
