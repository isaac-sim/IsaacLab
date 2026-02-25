# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import logging
import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import torch
import warp as wp

import omni.physics.tensors.impl.api as physx
from pxr import UsdShade

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets.asset_base import AssetBase
from isaaclab.markers import VisualizationMarkers

from isaaclab_physx.physics import PhysxManager as SimulationManager

from .deformable_object_data import DeformableObjectData
from .kernels import (
    compute_nodal_state_w,
    set_kinematic_flags_to_one,
    vec6f,
    write_nodal_vec3f_to_buffer,
    write_nodal_vec4f_to_buffer,
)

if TYPE_CHECKING:
    from .deformable_object_cfg import DeformableObjectCfg

# import logger
logger = logging.getLogger(__name__)


class DeformableObject(AssetBase):
    """A deformable object asset class.

    Deformable objects are assets that can be deformed in the simulation. They are typically used for
    soft bodies, such as stuffed animals and food items.

    Unlike rigid object assets, deformable objects have a more complex structure and require additional
    handling for simulation. The simulation of deformable objects follows a finite element approach, where
    the object is discretized into a mesh of nodes and elements. The nodes are connected by elements, which
    define the material properties of the object. The nodes can be moved and deformed, and the elements
    respond to these changes.

    The state of a deformable object comprises of its nodal positions and velocities, and not the object's root
    position and orientation. The nodal positions and velocities are in the simulation frame.

    Soft bodies can be `partially kinematic`_, where some nodes are driven by kinematic targets, and the rest are
    simulated. The kinematic targets are the desired positions of the nodes, and the simulation drives the nodes
    towards these targets. This is useful for partial control of the object, such as moving a stuffed animal's
    head while the rest of the body is simulated.

    .. attention::
        This class is experimental and subject to change due to changes on the underlying PhysX API on which
        it depends. We will try to maintain backward compatibility as much as possible but some changes may be
        necessary.

    .. _partially kinematic: https://nvidia-omniverse.github.io/PhysX/physx/5.4.1/docs/SoftBodies.html#kinematic-soft-bodies
    """

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
        return self.root_view.count

    @property
    def num_bodies(self) -> int:
        """Number of bodies in the asset.

        This is always 1 since each object is a single deformable body.
        """
        return 1

    @property
    def root_view(self) -> physx.SoftBodyView:
        """Deformable body view for the asset.

        .. note::
            Use this view with caution. It requires handling of tensors in a specific way.
        """
        return self._root_physx_view

    @property
    def root_physx_view(self) -> physx.SoftBodyView:
        """Deprecated property. Please use :attr:`root_view` instead."""
        logger.warning(
            "The `root_physx_view` property will be deprecated in a future release. Please use `root_view` instead."
        )
        return self.root_view

    @property
    def material_physx_view(self) -> physx.SoftBodyMaterialView | None:
        """Deformable material view for the asset (PhysX).

        This view is optional and may not be available if the material is not bound to the deformable body.
        If the material is not available, then the material properties will be set to default values.

        .. note::
            Use this view with caution. It requires handling of tensors in a specific way.
        """
        return self._material_physx_view

    @property
    def max_sim_elements_per_body(self) -> int:
        """The maximum number of simulation mesh elements per deformable body."""
        return self.root_view.max_sim_elements_per_body

    @property
    def max_collision_elements_per_body(self) -> int:
        """The maximum number of collision mesh elements per deformable body."""
        return self.root_view.max_elements_per_body

    @property
    def max_sim_vertices_per_body(self) -> int:
        """The maximum number of simulation mesh vertices per deformable body."""
        return self.root_view.max_sim_vertices_per_body

    @property
    def max_collision_vertices_per_body(self) -> int:
        """The maximum number of collision mesh vertices per deformable body."""
        return self.root_view.max_vertices_per_body

    """
    Operations.
    """

    def reset(self, env_ids: Sequence[int] | None = None, env_mask: wp.array | None = None) -> None:
        """Reset the deformable object.

        Args:
            env_ids: Environment indices. If None, then all indices are used.
            env_mask: Environment mask. If None, then all the instances are updated. Shape is (num_instances,).
        """
        # TODO: Should we reset the kinematic targets when resetting the object?
        #  This is not done in the current implementation. We assume users will reset the kinematic targets.
        pass

    def write_data_to_sim(self):
        pass

    def update(self, dt: float):
        self._data.update(dt)

    """
    Operations - Write to simulation.
    """

    def write_nodal_state_to_sim_index(
        self,
        nodal_state: torch.Tensor | wp.array,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        full_data: bool = False,
    ) -> None:
        """Set the nodal state over selected environment indices into the simulation.

        The nodal state comprises of the nodal positions and velocities. Since these are nodes, the velocity only has
        a translational component. All the quantities are in the simulation frame.

        Args:
            nodal_state: Nodal state in simulation frame.
                Shape is (len(env_ids), max_sim_vertices_per_body, 6) or (num_instances, max_sim_vertices_per_body, 6).
            env_ids: Environment indices. If None, then all indices are used.
            full_data: Whether to expect full data. Defaults to False.
        """
        # Convert warp to torch if needed
        if isinstance(nodal_state, wp.array):
            nodal_state = wp.to_torch(nodal_state)
        # set into simulation
        self.write_nodal_pos_to_sim_index(nodal_state[..., :3], env_ids=env_ids, full_data=full_data)
        self.write_nodal_velocity_to_sim_index(nodal_state[..., 3:], env_ids=env_ids, full_data=full_data)

    def write_nodal_state_to_sim_mask(
        self,
        nodal_state: torch.Tensor | wp.array,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set the nodal state over selected environment mask into the simulation.

        The nodal state comprises of the nodal positions and velocities. Since these are nodes, the velocity only has
        a translational component. All the quantities are in the simulation frame.

        Args:
            nodal_state: Nodal state in simulation frame.
                Shape is (num_instances, max_sim_vertices_per_body, 6).
            env_mask: Environment mask. If None, then all indices are used.
        """
        if env_mask is not None:
            env_ids = wp.nonzero(env_mask)
        else:
            env_ids = self._ALL_INDICES
        self.write_nodal_state_to_sim_index(nodal_state, env_ids=env_ids, full_data=True)

    def write_nodal_pos_to_sim_index(
        self,
        nodal_pos: torch.Tensor | wp.array,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        full_data: bool = False,
    ) -> None:
        """Set the nodal positions over selected environment indices into the simulation.

        The nodal position comprises of individual nodal positions of the simulation mesh for the deformable body.
        The positions are in the simulation frame.

        Args:
            nodal_pos: Nodal positions in simulation frame.
                Shape is (len(env_ids), max_sim_vertices_per_body, 3) or (num_instances, max_sim_vertices_per_body, 3).
            env_ids: Environment indices. If None, then all indices are used.
            full_data: Whether to expect full data. Defaults to False.
        """
        # resolve env_ids
        env_ids = self._resolve_env_ids(env_ids)
        # convert torch to warp if needed
        if isinstance(nodal_pos, torch.Tensor):
            nodal_pos = wp.from_torch(nodal_pos.contiguous(), dtype=wp.vec3f)
        # write into internal buffer via kernel
        wp.launch(
            write_nodal_vec3f_to_buffer,
            dim=(env_ids.shape[0], self.max_sim_vertices_per_body),
            inputs=[nodal_pos, env_ids, full_data],
            outputs=[self._data._nodal_pos_w.data],
            device=self.device,
        )
        # update timestamp
        self._data._nodal_pos_w.timestamp = self._data._sim_timestamp
        # invalidate dependent buffers
        self._data._nodal_state_w.timestamp = -1.0
        self._data._root_pos_w.timestamp = -1.0
        # set into simulation
        self.root_view.set_sim_nodal_positions(self._data._nodal_pos_w.data.view(wp.float32), indices=env_ids)

    def write_nodal_pos_to_sim_mask(
        self,
        nodal_pos: torch.Tensor | wp.array,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set the nodal positions over selected environment mask into the simulation.

        The nodal position comprises of individual nodal positions of the simulation mesh for the deformable body.
        The positions are in the simulation frame.

        Args:
            nodal_pos: Nodal positions in simulation frame.
                Shape is (num_instances, max_sim_vertices_per_body, 3).
            env_mask: Environment mask. If None, then all indices are used.
        """
        if env_mask is not None:
            env_ids = wp.nonzero(env_mask)
        else:
            env_ids = self._ALL_INDICES
        self.write_nodal_pos_to_sim_index(nodal_pos, env_ids=env_ids, full_data=True)

    def write_nodal_velocity_to_sim_index(
        self,
        nodal_vel: torch.Tensor | wp.array,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        full_data: bool = False,
    ) -> None:
        """Set the nodal velocity over selected environment indices into the simulation.

        The nodal velocity comprises of individual nodal velocities of the simulation mesh for the deformable
        body. Since these are nodes, the velocity only has a translational component. The velocities are in the
        simulation frame.

        Args:
            nodal_vel: Nodal velocities in simulation frame.
                Shape is (len(env_ids), max_sim_vertices_per_body, 3) or (num_instances, max_sim_vertices_per_body, 3).
            env_ids: Environment indices. If None, then all indices are used.
            full_data: Whether to expect full data. Defaults to False.
        """
        # resolve env_ids
        env_ids = self._resolve_env_ids(env_ids)
        # convert torch to warp if needed
        if isinstance(nodal_vel, torch.Tensor):
            nodal_vel = wp.from_torch(nodal_vel.contiguous(), dtype=wp.vec3f)
        # write into internal buffer via kernel
        wp.launch(
            write_nodal_vec3f_to_buffer,
            dim=(env_ids.shape[0], self.max_sim_vertices_per_body),
            inputs=[nodal_vel, env_ids, full_data],
            outputs=[self._data._nodal_vel_w.data],
            device=self.device,
        )
        # update timestamp
        self._data._nodal_vel_w.timestamp = self._data._sim_timestamp
        # invalidate dependent buffers
        self._data._nodal_state_w.timestamp = -1.0
        self._data._root_vel_w.timestamp = -1.0
        # set into simulation
        self.root_view.set_sim_nodal_velocities(self._data._nodal_vel_w.data.view(wp.float32), indices=env_ids)

    def write_nodal_velocity_to_sim_mask(
        self,
        nodal_vel: torch.Tensor | wp.array,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set the nodal velocity over selected environment mask into the simulation.

        The nodal velocity comprises of individual nodal velocities of the simulation mesh for the deformable
        body. Since these are nodes, the velocity only has a translational component. The velocities are in the
        simulation frame.

        Args:
            nodal_vel: Nodal velocities in simulation frame.
                Shape is (num_instances, max_sim_vertices_per_body, 3).
            env_mask: Environment mask. If None, then all indices are used.
        """
        if env_mask is not None:
            env_ids = wp.nonzero(env_mask)
        else:
            env_ids = self._ALL_INDICES
        self.write_nodal_velocity_to_sim_index(nodal_vel, env_ids=env_ids, full_data=True)

    def write_nodal_kinematic_target_to_sim_index(
        self,
        targets: torch.Tensor | wp.array,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        full_data: bool = False,
    ) -> None:
        """Set the kinematic targets of the simulation mesh for the deformable bodies using indices.

        The kinematic targets comprise of individual nodal positions of the simulation mesh for the deformable body
        and a flag indicating whether the node is kinematically driven or not. The positions are in the simulation
        frame.

        .. note::
            The flag is set to 0.0 for kinematically driven nodes and 1.0 for free nodes.

        Args:
            targets: The kinematic targets comprising of nodal positions and flags.
                Shape is (len(env_ids), max_sim_vertices_per_body, 4) or (num_instances, max_sim_vertices_per_body, 4).
            env_ids: Environment indices. If None, then all indices are used.
            full_data: Whether to expect full data. Defaults to False.
        """
        # resolve env_ids
        env_ids = self._resolve_env_ids(env_ids)
        # convert torch to warp if needed, ensuring 2D (num_envs, V, 4) -> (num_envs, V) vec4f
        if isinstance(targets, torch.Tensor):
            if targets.dim() == 2:
                targets = targets.unsqueeze(0)
            targets = wp.from_torch(targets.contiguous(), dtype=wp.vec4f)
        # write into internal buffer via kernel
        wp.launch(
            write_nodal_vec4f_to_buffer,
            dim=(env_ids.shape[0], self.max_sim_vertices_per_body),
            inputs=[targets, env_ids, full_data],
            outputs=[self._data.nodal_kinematic_target],
            device=self.device,
        )
        # set into simulation
        self.root_view.set_sim_kinematic_targets(self._data.nodal_kinematic_target.view(wp.float32), indices=env_ids)

    def write_nodal_kinematic_target_to_sim_mask(
        self,
        targets: torch.Tensor | wp.array,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set the kinematic targets of the simulation mesh for the deformable bodies using mask.

        The kinematic targets comprise of individual nodal positions of the simulation mesh for the deformable body
        and a flag indicating whether the node is kinematically driven or not. The positions are in the simulation
        frame.

        .. note::
            The flag is set to 0.0 for kinematically driven nodes and 1.0 for free nodes.

        Args:
            targets: The kinematic targets comprising of nodal positions and flags.
                Shape is (num_instances, max_sim_vertices_per_body, 4).
            env_mask: Environment mask. If None, then all indices are used.
        """
        if env_mask is not None:
            env_ids = wp.nonzero(env_mask)
        else:
            env_ids = self._ALL_INDICES
        self.write_nodal_kinematic_target_to_sim_index(targets, env_ids=env_ids, full_data=True)

    """
    Operations - Deprecated wrappers.
    """

    def write_nodal_state_to_sim(
        self,
        nodal_state: torch.Tensor | wp.array,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Deprecated. Please use :meth:`write_nodal_state_to_sim_index` instead."""
        warnings.warn(
            "The method 'write_nodal_state_to_sim' is deprecated. Please use 'write_nodal_state_to_sim_index' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.write_nodal_state_to_sim_index(nodal_state, env_ids=env_ids)

    def write_nodal_kinematic_target_to_sim(
        self,
        targets: torch.Tensor | wp.array,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Deprecated. Please use :meth:`write_nodal_kinematic_target_to_sim_index` instead."""
        warnings.warn(
            "The method 'write_nodal_kinematic_target_to_sim' is deprecated."
            " Please use 'write_nodal_kinematic_target_to_sim_index' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.write_nodal_kinematic_target_to_sim_index(targets, env_ids=env_ids)

    def write_nodal_pos_to_sim(
        self,
        nodal_pos: torch.Tensor | wp.array,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Deprecated. Please use :meth:`write_nodal_pos_to_sim_index` instead."""
        warnings.warn(
            "The method 'write_nodal_pos_to_sim' is deprecated. Please use 'write_nodal_pos_to_sim_index' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.write_nodal_pos_to_sim_index(nodal_pos, env_ids=env_ids)

    def write_nodal_velocity_to_sim(
        self,
        nodal_vel: torch.Tensor | wp.array,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Deprecated. Please use :meth:`write_nodal_velocity_to_sim_index` instead."""
        warnings.warn(
            "The method 'write_nodal_velocity_to_sim' is deprecated."
            " Please use 'write_nodal_velocity_to_sim_index' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.write_nodal_velocity_to_sim_index(nodal_vel, env_ids=env_ids)

    """
    Operations - Helper.
    """

    def transform_nodal_pos(
        self, nodal_pos: torch.tensor, pos: torch.Tensor | None = None, quat: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Transform the nodal positions based on the pose transformation.

        This function computes the transformation of the nodal positions based on the pose transformation.
        It multiplies the nodal positions with the rotation matrix of the pose and adds the translation.
        Internally, it calls the :meth:`isaaclab.utils.math.transform_points` function.

        Args:
            nodal_pos: The nodal positions in the simulation frame. Shape is (N, max_sim_vertices_per_body, 3).
            pos: The position transformation. Shape is (N, 3).
                Defaults to None, in which case the position is assumed to be zero.
            quat: The orientation transformation as quaternion (x, y, z, w). Shape is (N, 4).
                Defaults to None, in which case the orientation is assumed to be identity.

        Returns:
            The transformed nodal positions. Shape is (N, max_sim_vertices_per_body, 3).
        """
        # offset the nodal positions to center them around the origin
        mean_nodal_pos = nodal_pos.mean(dim=1, keepdim=True)
        nodal_pos = nodal_pos - mean_nodal_pos
        # transform the nodal positions based on the pose around the origin
        return math_utils.transform_points(nodal_pos, pos, quat) + mean_nodal_pos

    """
    Internal helper.
    """

    def _resolve_env_ids(self, env_ids):
        """Resolve environment indices to a warp int32 array."""
        if env_ids is None or (isinstance(env_ids, slice) and env_ids == slice(None)):
            return self._ALL_INDICES
        elif isinstance(env_ids, list):
            return wp.array(env_ids, dtype=wp.int32, device=self.device)
        elif isinstance(env_ids, torch.Tensor):
            return wp.from_torch(env_ids.to(torch.int32), dtype=wp.int32)
        return env_ids

    def _initialize_impl(self):
        # obtain global simulation view
        self._physics_sim_view = SimulationManager.get_physics_sim_view()
        # obtain the first prim in the regex expression (all others are assumed to be a copy of this)
        template_prim = sim_utils.find_first_matching_prim(self.cfg.prim_path)
        if template_prim is None:
            raise RuntimeError(f"Failed to find prim for expression: '{self.cfg.prim_path}'.")
        template_prim_path = template_prim.GetPath().pathString

        # find deformable root prims
        root_prims = sim_utils.get_all_matching_child_prims(
            template_prim_path,
            predicate=lambda prim: "PhysxDeformableBodyAPI" in prim.GetAppliedSchemas(),
            traverse_instance_prims=False,
        )
        if len(root_prims) == 0:
            raise RuntimeError(
                f"Failed to find a deformable body when resolving '{self.cfg.prim_path}'."
                " Please ensure that the prim has 'PhysxDeformableBodyAPI' applied."
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
                    if "PhysxDeformableBodyMaterialAPI" in mat_prim.GetAppliedSchemas():
                        material_prim = mat_prim
                        break
        if material_prim is None:
            logger.info(
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

        # Return if the asset is not found
        if self._root_physx_view._backend is None:
            raise RuntimeError(f"Failed to create deformable body at: {self.cfg.prim_path}. Please check PhysX logs.")

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
        logger.info(f"Deformable body initialized at: {root_prim_path_expr}")
        logger.info(f"Number of instances: {self.num_instances}")
        logger.info(f"Number of bodies: {self.num_bodies}")
        if self._material_physx_view is not None:
            logger.info(f"Deformable material initialized at: {material_prim_path_expr}")
            logger.info(f"Number of instances: {self._material_physx_view.count}")
        else:
            logger.info("No deformable material found. Material properties will be set to default values.")

        # container for data access
        self._data = DeformableObjectData(self.root_view, self.device)

        # create buffers
        self._create_buffers()
        # update the deformable body data
        self.update(0.0)

        # Initialize debug visualization handle
        if self._debug_vis_handle is None:
            # set initial state of debug visualization
            self.set_debug_vis(self.cfg.debug_vis)

    def _create_buffers(self):
        """Create buffers for storing data."""
        # constants
        self._ALL_INDICES = wp.array(np.arange(self.num_instances, dtype=np.int32), device=self.device)

        # default state
        # we use the initial nodal positions at spawn time as the default state
        # note: these are all in the simulation frame
        nodal_positions_raw = self.root_view.get_sim_nodal_positions()  # (N, V, 3) float32
        nodal_positions = nodal_positions_raw.view(wp.vec3f).reshape(
            (self.num_instances, self.max_sim_vertices_per_body)
        )
        nodal_velocities = wp.zeros(
            (self.num_instances, self.max_sim_vertices_per_body), dtype=wp.vec3f, device=self.device
        )
        # compute default nodal state as vec6f
        self._data.default_nodal_state_w = wp.zeros(
            (self.num_instances, self.max_sim_vertices_per_body), dtype=vec6f, device=self.device
        )
        wp.launch(
            compute_nodal_state_w,
            dim=(self.num_instances, self.max_sim_vertices_per_body),
            inputs=[nodal_positions, nodal_velocities],
            outputs=[self._data.default_nodal_state_w],
            device=self.device,
        )

        # kinematic targets — allocate our own buffer and copy from PhysX
        kinematic_raw = self.root_view.get_sim_kinematic_targets()  # (N, V, 4) float32
        kinematic_view = kinematic_raw.view(wp.vec4f).reshape((self.num_instances, self.max_sim_vertices_per_body))
        self._data.nodal_kinematic_target = wp.zeros(
            (self.num_instances, self.max_sim_vertices_per_body), dtype=wp.vec4f, device=self.device
        )
        wp.copy(self._data.nodal_kinematic_target, kinematic_view)
        # set all nodes as non-kinematic targets by default (flag = 1.0)
        wp.launch(
            set_kinematic_flags_to_one,
            dim=(self.num_instances * self.max_sim_vertices_per_body,),
            inputs=[self._data.nodal_kinematic_target.reshape((self.num_instances * self.max_sim_vertices_per_body,))],
            device=self.device,
        )

    """
    Internal simulation callbacks.
    """

    def _set_debug_vis_impl(self, debug_vis: bool):
        # set visibility of markers
        # note: parent only deals with callbacks. not their visibility
        if debug_vis:
            if not hasattr(self, "target_visualizer"):
                self.target_visualizer = VisualizationMarkers(self.cfg.visualizer_cfg)
            # set their visibility to true
            self.target_visualizer.set_visibility(True)
        else:
            if hasattr(self, "target_visualizer"):
                self.target_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # check where to visualize
        kinematic_target_torch = wp.to_torch(self.data.nodal_kinematic_target)
        targets_enabled = kinematic_target_torch[:, :, 3] == 0.0
        num_enabled = int(torch.sum(targets_enabled).item())
        # get positions if any targets are enabled
        if num_enabled == 0:
            # create a marker below the ground
            positions = torch.tensor([[0.0, 0.0, -10.0]], device=self.device)
        else:
            positions = kinematic_target_torch[targets_enabled][..., :3]
        # show target visualizer
        self.target_visualizer.visualize(positions)

    def _invalidate_initialize_callback(self, event):
        """Invalidates the scene elements."""
        # call parent
        super()._invalidate_initialize_callback(event)
        self._root_physx_view = None
