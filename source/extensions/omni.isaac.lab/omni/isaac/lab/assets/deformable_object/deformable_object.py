# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import carb
import omni.physics.tensors.impl.api as physx
from pxr import PhysxSchema, UsdShade

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
    def material_physx_view(self) -> physx.SoftBodyMaterialView | None:
        """Deformable material view for the asset (PhysX).

        This view is optional and may not be available if the material is not bound to the deformable body.
        If the material is not available, then the material properties will be set to default values.

        Note:
            Use this view with caution. It requires handling of tensors in a specific way.
        """
        return self._material_physx_view

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

        # find deformable root prims
        root_prims = sim_utils.get_all_matching_child_prims(
            template_prim_path, predicate=lambda prim: prim.HasAPI(PhysxSchema.PhysxDeformableBodyAPI)
        )
        if len(root_prims) == 0:
            raise RuntimeError(
                f"Failed to find a deformable body when resolving '{self.cfg.prim_path}'."
                " Please ensure that the prim has 'PhysxSchema.PhysxDeformableBodyAPI' applied."
            )
        if len(root_prims) > 1:
            raise RuntimeError(
                f"Failed to find a single deformable body when resolving '{self.cfg.prim_path}'."
                f" Found multiple '{root_prims}' under '{template_prim_path}'."
                " Please ensure that there is only one deformable body in the prim path tree."
            )
        # we only need the first one from the list
        root_prim = root_prims[0]

        # find deformable material prims
        material_prim = None
        # obtain material prim from the root prim
        # note: here we assume that all the root prims have their material prims at similar paths
        #   and we only need to find the first one. This may not be the case for all scenarios.
        #   However, the checks in that case get cumbersome and are not included here.
        if root_prim.HasAPI(UsdShade.MaterialBindingAPI):
            # check the materials that are bound with the purpose 'physics'
            material_paths = UsdShade.MaterialBindingAPI(root_prim).GetDirectBindingRel("physics").GetTargets()
            # iterate through targets and find the deformable body material
            if len(material_paths) > 0:
                for mat_path in material_paths:
                    mat_prim = root_prim.GetStage().GetPrimAtPath(mat_path)
                    if mat_prim.HasAPI(PhysxSchema.PhysxDeformableBodyMaterialAPI):
                        material_prim = mat_prim
                        break
        if material_prim is None:
            carb.log_warn(
                f"Failed to find a deformable material binding for '{root_prim.GetPath().pathString}'."
                " The material properties will be set to default values and are not modifiable at runtime."
                " If you want to modify the material properties, please ensure that the material is bound"
                " to the deformable body."
            )

        # resolve root path back into regex expression
        # -- root prim expression
        root_prim_path = root_prim.GetPath().pathString
        root_prim_path_expr = self.cfg.prim_path + root_prim_path[len(template_prim_path) :]
        # -- object view
        self._root_physx_view = self._physics_sim_view.create_soft_body_view(root_prim_path_expr.replace(".*", "*"))

        # resolve material path back into regex expression
        if material_prim is not None:
            # -- material prim expression
            material_prim_path = material_prim.GetPath().pathString
            # check if the material prim is under the template prim
            # if not then we are assuming that the single material prim is used for all the deformable bodies
            if template_prim_path in material_prim_path:
                material_prim_path_expr = self.cfg.prim_path + material_prim_path[len(template_prim_path) :]
            else:
                material_prim_path_expr = material_prim_path
            # -- material view
            self._material_physx_view = self._physics_sim_view.create_soft_body_material_view(
                material_prim_path_expr.replace(".*", "*")
            )
        else:
            self._material_physx_view = None

        # log information about the deformable body
        carb.log_info(f"Deformable body initialized at: {root_prim_path_expr}")
        carb.log_info(f"Number of instances: {self.num_instances}")
        carb.log_info(f"Number of bodies: {self.num_bodies}")
        if self._material_physx_view is not None:
            carb.log_info(f"Deformable material initialized at: {material_prim_path_expr}")
            carb.log_info(f"Number of instances: {self._material_physx_view.count}")
        else:
            carb.log_info("No deformable material found. Material properties will be set to default values.")

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
