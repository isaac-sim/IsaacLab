# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import logging
import re
import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import torch
import warp as wp

import omni.physics.tensors.impl.api as physx
from pxr import UsdPhysics

import isaaclab.sim as sim_utils
import isaaclab.utils.string as string_utils
from isaaclab.assets.rigid_object_collection.base_rigid_object_collection import BaseRigidObjectCollection
from isaaclab.utils.wrench_composer import WrenchComposer

from isaaclab_physx.assets import kernels as shared_kernels
from isaaclab_physx.physics import PhysxManager as SimulationManager

from .kernels import resolve_view_ids
from .rigid_object_collection_data import RigidObjectCollectionData

if TYPE_CHECKING:
    from isaaclab.assets.rigid_object_collection.rigid_object_collection_cfg import RigidObjectCollectionCfg

# import logger
logger = logging.getLogger(__name__)


class RigidObjectCollection(BaseRigidObjectCollection):
    """A rigid object collection class.

    This class represents a collection of rigid objects in the simulation, where the state of the
    rigid objects can be accessed and modified using a batched ``(env_ids, object_ids)`` API.

    For each rigid body in the collection, the root prim of the asset must have the `USD RigidBodyAPI`_
    applied to it. This API is used to define the simulation properties of the rigid bodies. On playing the
    simulation, the physics engine will automatically register the rigid bodies and create a corresponding
    rigid body handle. This handle can be accessed using the :attr:`root_view` attribute.

    Rigid objects in the collection are uniquely identified via the key of the dictionary
    :attr:`~isaaclab.assets.RigidObjectCollectionCfg.rigid_objects` in the
    :class:`~isaaclab.assets.RigidObjectCollectionCfg` configuration class.
    This differs from the :class:`~isaaclab.assets.RigidObject` class, where a rigid object is identified by
    the name of the Xform where the `USD RigidBodyAPI`_ is applied. This would not be possible for the rigid
    object collection since the :attr:`~isaaclab.assets.RigidObjectCollectionCfg.rigid_objects` dictionary
    could contain the same rigid object multiple times, leading to ambiguity.

    .. _`USD RigidBodyAPI`: https://openusd.org/dev/api/class_usd_physics_rigid_body_a_p_i.html
    """

    cfg: RigidObjectCollectionCfg
    """Configuration instance for the rigid object."""

    __backend_name__: str = "physx"
    """The name of the backend for the rigid object."""

    def __init__(self, cfg: RigidObjectCollectionCfg):
        """Initialize the rigid object.

        Args:
            cfg: A configuration instance.
        """
        # Note: We never call the parent constructor as it tries to call its own spawning which we don't want.
        # check that the config is valid
        cfg.validate()
        # store inputs
        self.cfg = cfg.copy()
        # flag for whether the asset is initialized
        self._is_initialized = False
        # spawn the rigid objects
        for rigid_body_cfg in self.cfg.rigid_objects.values():
            # spawn the asset
            if rigid_body_cfg.spawn is not None:
                rigid_body_cfg.spawn.func(
                    rigid_body_cfg.prim_path,
                    rigid_body_cfg.spawn,
                    translation=rigid_body_cfg.init_state.pos,
                    orientation=rigid_body_cfg.init_state.rot,
                )
            # check that spawn was successful
            matching_prims = sim_utils.find_matching_prims(rigid_body_cfg.prim_path)
            if len(matching_prims) == 0:
                raise RuntimeError(f"Could not find prim with path {rigid_body_cfg.prim_path}.")
        # stores object names
        self._body_names_list = []

        # register various callback functions
        self._register_callbacks()
        self._debug_vis_handle = None

    """
    Properties
    """

    @property
    def data(self) -> RigidObjectCollectionData:
        return self._data

    @property
    def num_instances(self) -> int:
        return self.root_view.count // self.num_bodies

    @property
    def num_bodies(self) -> int:
        """Number of bodies in the rigid object collection."""
        return len(self.body_names)

    @property
    def body_names(self) -> list[str]:
        """Ordered names of bodies in the rigid object collection."""
        return self._body_names_list

    @property
    def root_view(self):
        """Root view for the rigid object collection.

        .. note::
            Use this view with caution. It requires handling of tensors in a specific way.
        """
        return self._root_view

    @property
    def instantaneous_wrench_composer(self) -> WrenchComposer:
        """Instantaneous wrench composer for the rigid object collection."""
        return self._instantaneous_wrench_composer

    @property
    def permanent_wrench_composer(self) -> WrenchComposer:
        """Permanent wrench composer for the rigid object collection."""
        return self._permanent_wrench_composer

    """
    Operations.
    """

    def reset(self, env_ids: torch.Tensor | None = None, object_ids: slice | torch.Tensor | None = None) -> None:
        """Resets all internal buffers of selected environments and objects.

        Args:
            env_ids: Environment indices. If None, then all indices are used.
            object_ids: Object indices. If None, then all indices are used.
        """
        # resolve all indices
        if env_ids is None:
            env_ids = self._ALL_ENV_INDICES
        if object_ids is None:
            object_ids = self._ALL_BODY_INDICES
        # reset external wrench
        self._instantaneous_wrench_composer.reset(env_ids)
        self._permanent_wrench_composer.reset(env_ids)

    def write_data_to_sim(self) -> None:
        """Write external wrench to the simulation.

        .. note::
            We write external wrench to the simulation here since this function is called before the simulation step.
            This ensures that the external wrench is applied at every simulation step.
        """
        # write external wrench
        if self._instantaneous_wrench_composer.active or self._permanent_wrench_composer.active:
            if self._instantaneous_wrench_composer.active:
                # Compose instantaneous wrench with permanent wrench
                self._instantaneous_wrench_composer.add_forces_and_torques(
                    forces=self._permanent_wrench_composer.composed_force,
                    torques=self._permanent_wrench_composer.composed_torque,
                    body_ids=self._ALL_BODY_INDICES,
                    env_ids=self._ALL_ENV_INDICES,
                )
                # Apply both instantaneous and permanent wrench to the simulation
                self.root_view.apply_forces_and_torques_at_position(
                    force_data=self.reshape_data_to_view_2d(
                        self._instantaneous_wrench_composer.composed_force, device=self.device
                    ).view(wp.float32),
                    torque_data=self.reshape_data_to_view_2d(
                        self._instantaneous_wrench_composer.composed_torque, device=self.device
                    ).view(wp.float32),
                    position_data=None,
                    indices=self._env_body_ids_to_view_ids(
                        self._ALL_ENV_INDICES, self._ALL_BODY_INDICES, device=self.device
                    ),
                    is_global=False,
                )
            else:
                # Apply permanent wrench to the simulation
                self.root_view.apply_forces_and_torques_at_position(
                    force_data=self.reshape_data_to_view_2d(
                        self._permanent_wrench_composer.composed_force, device=self.device
                    ).view(wp.float32),
                    torque_data=self.reshape_data_to_view_2d(
                        self._permanent_wrench_composer.composed_torque, device=self.device
                    ).view(wp.float32),
                    position_data=None,
                    indices=self._env_body_ids_to_view_ids(
                        self._ALL_ENV_INDICES, self._ALL_BODY_INDICES, device=self.device
                    ),
                    is_global=False,
                )
        self._instantaneous_wrench_composer.reset()

    def update(self, dt: float) -> None:
        """Updates the simulation data.

        Args:
            dt: The time step size in seconds.
        """
        self.data.update(dt)

    """
    Operations - Finders.
    """

    def find_bodies(
        self, name_keys: str | Sequence[str], preserve_order: bool = False
    ) -> tuple[torch.Tensor, list[str]]:
        """Find bodies in the rigid body collection based on the name keys.

        Please check the :meth:`isaaclab.utils.string_utils.resolve_matching_names` function for more
        information on the name matching.

        Args:
            name_keys: A regular expression or a list of regular expressions to match the body names.
            preserve_order: Whether to preserve the order of the name keys in the output. Defaults to False.

        Returns:
            A tuple of lists containing the body indices and names.
        """
        obj_ids, obj_names = string_utils.resolve_matching_names(name_keys, self.object_names, preserve_order)
        return torch.tensor(obj_ids, device=self.device, dtype=torch.int32), obj_names

    """
    Operations - Write to simulation.
    """

    def write_body_pose_to_sim_index(
        self,
        body_poses: torch.Tensor | wp.array,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        body_ids: Sequence[int] | torch.Tensor | wp.array | slice | None = None,
    ) -> None:
        """Set the body pose over selected environment and body indices into the simulation.

        The body pose comprises of the cartesian position and quaternion orientation in (x, y, z, w).

        .. note::
            This method expects partial data.

        .. tip::
            For maximum performance we recommend looking at the actual implementation of the method in the backend.
            Some backends may provide optimized implementations for masks / indices.

        Args:
            body_poses: Body poses in simulation frame. Shape is (len(env_ids), len(body_ids), 7).
            env_ids: Environment indices. If None, then all indices are used.
            body_ids: Body indices. If None, then all indices are used.
            full_data: Whether to expect full data. Defaults to False.
        """
        self.write_body_link_pose_to_sim_index(body_poses, env_ids=env_ids, body_ids=body_ids)

    def write_body_pose_to_sim_mask(
        self,
        body_poses: torch.Tensor | wp.array,
        env_mask: wp.array | None = None,
        body_mask: wp.array | None = None,
    ) -> None:
        """Set the body pose over selected environment mask into the simulation.

        The body pose comprises of the cartesian position and quaternion orientation in (x, y, z, w).

        .. note::
            This method expects full data.

        .. tip::
            For maximum performance we recommend looking at the actual implementation of the method in the backend.

        Args:
            body_poses: Body poses in simulation frame. Shape is (num_instances, num_bodies, 7).
            env_mask: Environment mask. If None, then all indices are used.
            body_mask: Body mask. If None, then all bodies are used.
        """
        if env_mask is not None:
            env_ids = wp.nonzero(env_mask)
        else:
            env_ids = self._ALL_ENV_INDICES
        if body_mask is not None:
            body_ids = wp.nonzero(body_mask)
        else:
            body_ids = self._ALL_BODY_INDICES
        self.write_body_link_pose_to_sim_index(body_poses, env_ids=env_ids, body_ids=body_ids, full_data=True)

    def write_body_velocity_to_sim_index(
        self,
        body_velocities: torch.Tensor | wp.array,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        body_ids: Sequence[int] | torch.Tensor | wp.array | slice | None = None,
    ) -> None:
        """Set the body velocity over selected environment and body indices into the simulation.

        The velocity comprises linear velocity (x, y, z) and angular velocity (x, y, z) in that order.

        .. note::
            This method expects partial data.

        .. tip::
            For maximum performance we recommend looking at the actual implementation of the method in the backend.
            Some backends may provide optimized implementations for masks / indices.

        Args:
            body_velocities: Body velocities in simulation frame.
                Shape is (len(env_ids), len(body_ids), 6) or (num_instances, num_bodies, 6).
            env_ids: Environment indices. If None, then all indices are used.
            body_ids: Body indices. If None, then all indices are used.
        """
        self.write_body_com_velocity_to_sim_index(body_velocities, env_ids=env_ids, body_ids=body_ids)

    def write_body_velocity_to_sim_mask(
        self,
        body_velocities: torch.Tensor | wp.array,
        env_mask: wp.array | None = None,
        body_mask: wp.array | None = None,
    ) -> None:
        """Set the body velocity over selected environment mask into the simulation.

        The velocity comprises linear velocity (x, y, z) and angular velocity (x, y, z) in that order.

        .. note::
            This method expects full data.

        .. tip::
            For maximum performance we recommend looking at the actual implementation of the method in the backend.
            Some backends may provide optimized implementations for masks / indices.

        Args:
            body_velocities: Body velocities in simulation frame.
                Shape is (num_instances, num_bodies, 6).
            env_mask: Environment mask. If None, then all indices are used.
            body_mask: Body mask. If None, then all bodies are used.
        """
        if env_mask is not None:
            env_ids = wp.nonzero(env_mask)
        else:
            env_ids = self._ALL_ENV_INDICES
        if body_mask is not None:
            body_ids = wp.nonzero(body_mask)
        else:
            body_ids = self._ALL_BODY_INDICES
        self.write_body_com_velocity_to_sim_index(body_velocities, env_ids=env_ids, body_ids=body_ids, full_data=True)

    def write_body_link_pose_to_sim_index(
        self,
        body_poses: torch.Tensor | wp.array,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        body_ids: Sequence[int] | torch.Tensor | wp.array | slice | None = None,
        full_data: bool = False,
    ) -> None:
        """Set the body link pose over selected environment and body indices into the simulation.

        Args:
            body_poses: Body link poses in simulation frame.
                Shape is (len(env_ids), len(body_ids), 7) or (num_instances, num_bodies, 7).
            env_ids: Environment indices. If None, then all indices are used.
            body_ids: Body indices. If None, then all indices are used.
            full_data: Whether to expect full data. Defaults to False.
        """
        env_ids = self._resolve_env_ids(env_ids)
        body_ids = self._resolve_body_ids(body_ids)
        wp.launch(
            shared_kernels.set_body_link_pose_to_sim,
            dim=(env_ids.shape[0], body_ids.shape[0]),
            inputs=[
                body_poses,
                env_ids,
                body_ids,
                full_data,
            ],
            outputs=[
                self.data._body_link_pose_w.data,
                None,  # self.data._body_link_state_w.data,
                None,  # self.data._body_state_w.data,
            ],
            device=self.device,
        )
        # Update the timestamps
        self.data._body_link_pose_w.timestamp = self.data._sim_timestamp
        # Invalidate dependent timestamps
        self.data._body_com_pose_w.timestamp = -1.0
        self.data._body_com_state_w.timestamp = -1.0
        self.data._body_link_state_w.timestamp = -1.0
        self.data._body_state_w.timestamp = -1.0
        # set into simulation
        view_ids = self._env_body_ids_to_view_ids(env_ids, body_ids, device=self.device)
        self.root_view.set_transforms(
            self.reshape_data_to_view_2d(self.data._body_link_pose_w.data, device=self.device).view(wp.float32),
            indices=view_ids,
        )

    def write_body_link_pose_to_sim_mask(
        self,
        body_poses: torch.Tensor | wp.array,
        env_mask: wp.array | None = None,
        body_ids: Sequence[int] | torch.Tensor | wp.array | slice | None = None,
    ) -> None:
        """Set the body link pose over selected environment mask into the simulation.

        Args:
            body_poses: Body link poses in simulation frame. Shape is (num_instances, num_bodies, 7).
            env_mask: Environment mask. If None, then all indices are used.
            body_ids: Body indices. If None, then all indices are used.
        """
        if env_mask is not None:
            env_ids = wp.nonzero(env_mask)
        else:
            env_ids = self._ALL_ENV_INDICES
        self.write_body_link_pose_to_sim_index(body_poses, env_ids=env_ids, body_ids=body_ids, full_data=True)

    def write_body_com_pose_to_sim_index(
        self,
        body_poses: torch.Tensor | wp.array,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        body_ids: Sequence[int] | torch.Tensor | wp.array | slice | None = None,
        full_data: bool = False,
    ) -> None:
        """Set the body center of mass pose over selected environment and body indices into the simulation.

        Args:
            body_poses: Body center of mass poses in simulation frame.
                Shape is (len(env_ids), len(body_ids), 7) or (num_instances, num_bodies, 7).
            env_ids: Environment indices. If None, then all indices are used.
            body_ids: Body indices. If None, then all indices are used.
            full_data: Whether to expect full data. Defaults to False.
        """
        env_ids = self._resolve_env_ids(env_ids)
        body_ids = self._resolve_body_ids(body_ids)
        wp.launch(
            shared_kernels.set_body_com_pose_to_sim,
            dim=(env_ids.shape[0], body_ids.shape[0]),
            inputs=[
                body_poses,
                self.data.body_com_pose_b,
                env_ids,
                body_ids,
                full_data,
            ],
            outputs=[
                self.data._body_com_pose_w.data,
                self.data._body_link_pose_w.data,
                None,  # self.data._body_com_state_w.data,
                None,  # self.data._body_link_state_w.data,
                None,  # self.data._body_state_w.data,
            ],
            device=self.device,
        )
        # Update the timestamps
        self.data._body_com_pose_w.timestamp = self.data._sim_timestamp
        self.data._body_link_pose_w.timestamp = self.data._sim_timestamp
        # Invalidate dependent timestamps
        self.data._body_link_state_w.timestamp = -1.0
        self.data._body_state_w.timestamp = -1.0
        self.data._body_com_state_w.timestamp = -1.0
        # set into simulation
        view_ids = self._env_body_ids_to_view_ids(env_ids, body_ids, device=self.device)
        self.root_view.set_transforms(
            self.reshape_data_to_view_2d(self.data._body_link_pose_w.data, device=self.device).view(wp.float32),
            indices=view_ids,
        )

    def write_body_com_pose_to_sim_mask(
        self,
        body_poses: torch.Tensor | wp.array,
        env_mask: wp.array | None = None,
        body_ids: Sequence[int] | torch.Tensor | wp.array | slice | None = None,
    ) -> None:
        """Set the body center of mass pose over selected environment mask into the simulation.

        Args:
            body_poses: Body center of mass poses in simulation frame. Shape is (num_instances, num_bodies, 7).
            env_mask: Environment mask. If None, then all indices are used.
            body_ids: Body indices. If None, then all indices are used.
        """
        if env_mask is not None:
            env_ids = wp.nonzero(env_mask)
        else:
            env_ids = self._ALL_ENV_INDICES
        self.write_body_com_pose_to_sim_index(body_poses, env_ids=env_ids, body_ids=body_ids, full_data=True)

    def write_body_com_velocity_to_sim_index(
        self,
        body_velocities: torch.Tensor | wp.array,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        body_ids: Sequence[int] | torch.Tensor | wp.array | slice | None = None,
        full_data: bool = False,
    ) -> None:
        """Set the body center of mass velocity over selected environment and body indices into the simulation.

        Args:
            body_velocities: Body center of mass velocities in simulation frame.
                Shape is (len(env_ids), len(body_ids), 6) or (num_instances, num_bodies, 6).
            env_ids: Environment indices. If None, then all indices are used.
            body_ids: Body indices. If None, then all indices are used.
            full_data: Whether to expect full data. Defaults to False.
        """
        env_ids = self._resolve_env_ids(env_ids)
        body_ids = self._resolve_body_ids(body_ids)
        wp.launch(
            shared_kernels.set_body_com_velocity_to_sim,
            dim=(env_ids.shape[0], body_ids.shape[0]),
            inputs=[
                body_velocities,
                env_ids,
                body_ids,
                full_data,
            ],
            outputs=[
                self.data._body_com_vel_w.data,
                self.data._body_com_acc_w.data,
                None,  # self.data._body_state_w.data,
                None,  # self.data._body_com_state_w.data,
            ],
            device=self.device,
        )
        # Update the timestamps
        self.data._body_com_vel_w.timestamp = self.data._sim_timestamp
        self.data._body_com_acc_w.timestamp = self.data._sim_timestamp
        # Invalidate dependent timestamps
        self.data._body_link_vel_w.timestamp = -1.0
        self.data._body_state_w.timestamp = -1.0
        self.data._body_com_state_w.timestamp = -1.0
        self.data._body_link_state_w.timestamp = -1.0
        # set into simulation
        view_ids = self._env_body_ids_to_view_ids(env_ids, body_ids, device=self.device)
        self.root_view.set_velocities(
            self.reshape_data_to_view_2d(self.data._body_com_vel_w.data, device=self.device).view(wp.float32),
            indices=view_ids,
        )

    def write_body_com_velocity_to_sim_mask(
        self,
        body_velocities: torch.Tensor | wp.array,
        env_mask: wp.array | None = None,
        body_ids: Sequence[int] | torch.Tensor | wp.array | slice | None = None,
    ) -> None:
        """Set the body center of mass velocity over selected environment mask into the simulation.

        Args:
            body_velocities: Body center of mass velocities in simulation frame.
                Shape is (num_instances, num_bodies, 6).
            env_mask: Environment mask. If None, then all indices are used.
            body_ids: Body indices. If None, then all indices are used.
        """
        if env_mask is not None:
            env_ids = wp.nonzero(env_mask)
        else:
            env_ids = self._ALL_ENV_INDICES
        self.write_body_com_velocity_to_sim_index(body_velocities, env_ids=env_ids, body_ids=body_ids, full_data=True)

    def write_body_link_velocity_to_sim_index(
        self,
        body_velocities: torch.Tensor | wp.array,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        body_ids: Sequence[int] | torch.Tensor | wp.array | slice | None = None,
        full_data: bool = False,
    ) -> None:
        """Set the body link velocity over selected environment and body indices into the simulation.

        Args:
            body_velocities: Body link velocities in simulation frame.
                Shape is (len(env_ids), len(body_ids), 6) or (num_instances, num_bodies, 6).
            env_ids: Environment indices. If None, then all indices are used.
            body_ids: Body indices. If None, then all indices are used.
            full_data: Whether to expect full data. Defaults to False.
        """
        env_ids = self._resolve_env_ids(env_ids)
        body_ids = self._resolve_body_ids(body_ids)
        # Access body_com_pose_b and body_link_pose_w to ensure they are current.
        wp.launch(
            shared_kernels.set_body_link_velocity_to_sim,
            dim=(env_ids.shape[0], body_ids.shape[0]),
            inputs=[
                body_velocities,
                self.data.body_com_pose_b,
                self.data.body_link_pose_w,
                env_ids,
                body_ids,
                full_data,
            ],
            outputs=[
                self.data._body_link_vel_w.data,
                self.data._body_com_vel_w.data,
                self.data._body_com_acc_w.data,
                None,  # self.data._body_link_state_w.data,
                None,  # self.data._body_state_w.data,
                None,  # self.data._body_com_state_w.data,
            ],
            device=self.device,
        )
        # Update the timestamps
        self.data._body_link_vel_w.timestamp = self.data._sim_timestamp
        self.data._body_com_vel_w.timestamp = self.data._sim_timestamp
        self.data._body_com_acc_w.timestamp = self.data._sim_timestamp
        # Invalidate dependent timestamps
        self.data._body_link_state_w.timestamp = -1.0
        self.data._body_state_w.timestamp = -1.0
        self.data._body_com_state_w.timestamp = -1.0
        # set into simulation
        view_ids = self._env_body_ids_to_view_ids(env_ids, body_ids, device=self.device)
        self.root_view.set_velocities(
            self.reshape_data_to_view_2d(self.data._body_com_vel_w.data, device=self.device).view(wp.float32),
            indices=view_ids,
        )

    def write_body_link_velocity_to_sim_mask(
        self,
        body_velocities: torch.Tensor | wp.array,
        env_mask: wp.array | None = None,
        body_ids: Sequence[int] | torch.Tensor | wp.array | slice | None = None,
    ) -> None:
        """Set the body link velocity over selected environment mask into the simulation.

        Args:
            body_velocities: Body link velocities in simulation frame. Shape is (num_instances, num_bodies, 6).
            env_mask: Environment mask. If None, then all indices are used.
            body_ids: Body indices. If None, then all indices are used.
        """
        if env_mask is not None:
            env_ids = wp.nonzero(env_mask)
        else:
            env_ids = self._ALL_ENV_INDICES
        self.write_body_link_velocity_to_sim_index(body_velocities, env_ids=env_ids, body_ids=body_ids, full_data=True)

    """
    Operations - Setters.
    """

    def set_masses_index(
        self,
        masses: torch.Tensor | wp.array,
        body_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        full_data: bool = False,
    ) -> None:
        """Set masses of all bodies using indices.

        .. note::
            This method expects partial data or full data.

        .. tip::
            For maximum performance we recommend using the index method. This is because in PhysX, the tensor API
            is only supporting indexing, hence masks need to be converted to indices.

        Args:
            masses: Masses of all bodies. Shape is ``(len(env_ids), len(body_ids))``
                or ``(num_instances, num_bodies)`` if full_data.
            body_ids: The body indices to set the masses for. Defaults to None (all bodies).
            env_ids: The environment indices to set the masses for. Defaults to None (all environments).
            full_data: Whether to expect full data. Defaults to False.
        """
        # resolve all indices
        env_ids = self._resolve_env_ids(env_ids)
        body_ids = self._resolve_body_ids(body_ids)
        # Warp kernels can ingest torch tensors directly, so we don't need to convert to warp arrays here.
        wp.launch(
            shared_kernels.write_2d_data_to_buffer_with_indices,
            dim=(env_ids.shape[0], body_ids.shape[0]),
            inputs=[
                masses,
                env_ids,
                body_ids,
                full_data,
            ],
            outputs=[
                self.data._body_mass,
            ],
            device=self.device,
        )
        # Set into simulation, note that when updating "model" properties with PhysX we need to do it on CPU.
        # Convert from instance order (num_instances, num_bodies) to view order (num_bodies*num_instances, 1) for PhysX.
        mass_view_order = self.reshape_data_to_view_2d(self.data._body_mass, device="cpu")  # -> (B*I, 1)
        view_ids = self._env_body_ids_to_view_ids(env_ids, body_ids, device="cpu")
        self.root_view.set_masses(mass_view_order, indices=view_ids)

    def set_masses_mask(
        self,
        masses: torch.Tensor | wp.array,
        body_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set masses of all bodies using masks.

        .. note::
            This method expects full data.

        .. tip::
            For maximum performance we recommend using the index method. This is because in PhysX, the tensor API
            is only supporting indexing, hence masks need to be converted to indices.

        Args:
            masses: Masses of all bodies. Shape is ``(num_instances, num_bodies)``.
            body_mask: Body mask. If None, then all bodies are used.
            env_mask: Environment mask. If None, then all indices are used.
        """
        # Resolve masks.
        if env_mask is not None:
            env_ids = wp.nonzero(env_mask)
        else:
            env_ids = self._ALL_ENV_INDICES
        if body_mask is not None:
            body_ids = wp.nonzero(body_mask)
        else:
            body_ids = self._ALL_BODY_INDICES
        # Set full data to True to ensure the right code path is taken inside the kernel.
        self.set_masses_index(masses, body_ids=body_ids, env_ids=env_ids, full_data=True)

    def set_coms_index(
        self,
        coms: torch.Tensor | wp.array,
        body_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        full_data: bool = False,
    ) -> None:
        """Set center of mass pose of all bodies using indices.

        .. note::
            This method expects partial data or full data.

        .. tip::
            For maximum performance we recommend using the index method. This is because in PhysX, the tensor API
            is only supporting indexing, hence masks need to be converted to indices.

        Args:
            coms: Center of mass pose of all bodies. Shape is ``(len(env_ids), len(body_ids), 7)``
                or ``(num_instances, num_bodies, 7)`` if full_data.
            body_ids: The body indices to set the center of mass pose for. Defaults to None (all bodies).
            env_ids: The environment indices to set the center of mass pose for. Defaults to None (all environments).
            full_data: Whether to expect full data. Defaults to False.
        """
        # resolve all indices
        env_ids = self._resolve_env_ids(env_ids)
        body_ids = self._resolve_body_ids(body_ids)
        # Warp kernels can ingest torch tensors directly, so we don't need to convert to warp arrays here.
        wp.launch(
            shared_kernels.write_body_com_pose_to_buffer,
            dim=(env_ids.shape[0], body_ids.shape[0]),
            inputs=[
                coms,
                env_ids,
                body_ids,
                full_data,
            ],
            outputs=[
                self.data._body_com_pose_b.data,
            ],
            device=self.device,
        )
        # Invalidate the cached buffer
        self.data._body_com_pose_b.timestamp = self.data._sim_timestamp
        # Set into simulation, note that when updating "model" properties with PhysX we need to do it on CPU.
        # Convert from instance order (num_instances, num_bodies, 7) to view order (num_bodies*num_instances, 7) for
        # PhysX.
        com_view_order = self.reshape_data_to_view_2d(self.data._body_com_pose_b.data, device="cpu")  # (B*I, 7)
        view_ids = self._env_body_ids_to_view_ids(env_ids, body_ids, device="cpu")
        self.root_view.set_coms(com_view_order, indices=view_ids)

    def set_coms_mask(
        self,
        coms: torch.Tensor | wp.array,
        body_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set center of mass pose of all bodies using masks.

        .. note::
            This method expects full data.

        .. tip::
            For maximum performance we recommend using the index method. This is because in PhysX, the tensor API
            is only supporting indexing, hence masks need to be converted to indices.

        Args:
            coms: Center of mass pose of all bodies. Shape is ``(num_instances, num_bodies, 7)``.
            body_mask: Body mask. If None, then all bodies are used.
            env_mask: Environment mask. If None, then all indices are used.
        """
        # Resolve masks.
        if env_mask is not None:
            env_ids = wp.nonzero(env_mask)
        else:
            env_ids = self._ALL_ENV_INDICES
        if body_mask is not None:
            body_ids = wp.nonzero(body_mask)
        else:
            body_ids = self._ALL_BODY_INDICES
        # Set full data to True to ensure the right code path is taken inside the kernel.
        self.set_coms_index(coms, body_ids=body_ids, env_ids=env_ids, full_data=True)

    def set_inertias_index(
        self,
        inertias: torch.Tensor | wp.array,
        body_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        full_data: bool = False,
    ) -> None:
        """Set inertias of all bodies using indices.

        .. note::
            This method expects partial data or full data.

        .. tip::
            For maximum performance we recommend using the index method. This is because in PhysX, the tensor API
            is only supporting indexing, hence masks need to be converted to indices.

        Args:
            inertias: Inertias of all bodies. Shape is ``(len(env_ids), len(body_ids), 9)``
                or ``(num_instances, num_bodies, 9)`` if full_data.
            body_ids: The body indices to set the inertias for. Defaults to None (all bodies).
            env_ids: The environment indices to set the inertias for. Defaults to None (all environments).
            full_data: Whether to expect full data. Defaults to False.
        """
        # resolve all indices
        env_ids = self._resolve_env_ids(env_ids)
        body_ids = self._resolve_body_ids(body_ids)
        # Warp kernels can ingest torch tensors directly, so we don't need to convert to warp arrays here.
        wp.launch(
            shared_kernels.write_body_inertia_to_buffer,
            dim=(env_ids.shape[0], body_ids.shape[0]),
            inputs=[
                inertias,
                env_ids,
                body_ids,
                full_data,
            ],
            outputs=[
                self.data._body_inertia,
            ],
            device=self.device,
        )
        # Set into simulation, note that when updating "model" properties with PhysX we need to do it on CPU.
        # Convert from instance order (num_instances, num_bodies) to view order for PhysX.
        inertia_view_order = self.reshape_data_to_view_2d(self.data._body_inertia, device="cpu")
        view_ids = self._env_body_ids_to_view_ids(env_ids, body_ids, device="cpu")
        self.root_view.set_inertias(inertia_view_order, indices=view_ids)

    def set_inertias_mask(
        self,
        inertias: torch.Tensor | wp.array,
        body_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set inertias of all bodies using masks.

        .. note::
            This method expects full data.

        .. tip::
            For maximum performance we recommend using the index method. This is because in PhysX, the tensor API
            is only supporting indexing, hence masks need to be converted to indices.

        Args:
            inertias: Inertias of all bodies. Shape is ``(num_instances, num_bodies, 9)``.
            body_mask: Body mask. If None, then all bodies are used.
            env_mask: Environment mask. If None, then all indices are used.
        """
        # Resolve masks.
        if env_mask is not None:
            env_ids = wp.nonzero(env_mask)
        else:
            env_ids = self._ALL_ENV_INDICES
        if body_mask is not None:
            body_ids = wp.nonzero(body_mask)
        else:
            body_ids = self._ALL_BODY_INDICES
        # Set full data to True to ensure the right code path is taken inside the kernel.
        self.set_inertias_index(inertias, body_ids=body_ids, env_ids=env_ids, full_data=True)

    """
    Helper functions.
    """

    def reshape_view_to_data_2d(self, data: wp.array, device: str = "cpu") -> wp.array:
        """Reshapes and arranges the data from the physics view to (num_instances, num_bodies, data_size).

        The view returns data ordered as: ``(num_bodies * num_instances,)``
        ``[body0_env0, body0_env1, ..., body1_env0, body1_env1, ...]``

        This function returns the data arranged as::

            [[env_0_body_0, env_0_body_1, ...], [env_1_body_0, env_1_body_1, ...], ...]

        The shape of the returned data is ``(num_instances, num_bodies)``.

        Args:
            data: The data from the physics view. Shape is (num_instances * num_bodies).

        Returns:
            The reshaped data. Shape is (num_instances, num_bodies).
        """
        element_size = wp.types.type_size_in_bytes(data.dtype)
        strided_view = wp.array(
            ptr=data.ptr,
            shape=(self.num_instances, self.num_bodies),
            dtype=data.dtype,
            strides=(element_size, self.num_instances * element_size),
            device=self.device,
        )
        # Clone to make contiguous
        return wp.clone(strided_view, device=device)

    def reshape_view_to_data_3d(self, data: wp.array, data_dim: int, device: str = "cpu") -> wp.array:
        """Reshapes and arranges 3D view data to (num_instances, num_bodies, data_dim).

        The view returns data ordered as ``(num_bodies * num_instances, data_dim)``::

            [[body0_env0_data_0, body0_env0_data_1, ...], [body0_env1_data_0, body0_env1_data_1, ...], ...]

        This function returns the data arranged as ``(num_instances, num_bodies, data_dim)``::

            [
                [[env_0_body_0_data_0, env_0_body_0_data_1, ...], [env_0_body_1_data_0, env_0_body_1_data_1, ...], ...],
                [[env_1_body_0_data_0, env_1_body_0_data_1, ...], [env_1_body_1_data_0, env_1_body_1_data_1, ...], ...],
                ...,
            ]

        Args:
            data: The data from the physics view. Shape is (num_bodies * num_instances, data_dim).
            data_dim: The trailing dimension size.

        Returns:
            The reshaped data. Shape is (num_instances, num_bodies, data_dim).
        """
        element_size = wp.types.type_size_in_bytes(data.dtype)
        row_size = element_size * data_dim
        strided_view = wp.array(
            ptr=data.ptr,
            shape=(self.num_instances, self.num_bodies, data_dim),
            dtype=data.dtype,
            strides=(row_size, self.num_instances * row_size, element_size),
            device=self.device,
        )
        return wp.clone(strided_view, device=device)

    def reshape_data_to_view_2d(self, data: wp.array, device: str = "cpu") -> wp.array:
        """Reshapes and arranges the data to the be consistent with data from the :attr:`root_view`.

            Our internal methods consume and return data arranged as:
                [[env_0_body_0, env_0_body_1, ...],
                 [env_1_body_0, env_1_body_1, ...],
                 ...]
            The view needs data ordered as: (num_bodies * num_instances,)
                [body0_env0, body0_env1, ..., body1_env0, body1_env1, ...]

        Args:
            data: The data to be formatted for the view. Shape is (num_instances, num_bodies).

        Returns:
            The data formatted for the view. Shape is (num_bodies * num_instances,).
        """
        element_size = wp.types.type_size_in_bytes(data.dtype)
        strided_view = wp.array(
            ptr=data.ptr,
            shape=(self.num_bodies, self.num_instances),
            dtype=data.dtype,
            strides=(element_size, self.num_bodies * element_size),
            device=data.device,
        )
        # Clone to make contiguous (now row-major num_bodies x num_instances), then flatten
        return wp.clone(strided_view, device=device).reshape((self.num_bodies * self.num_instances,))

    def reshape_data_to_view_3d(self, data: wp.array, data_dim: int, device: str = "cpu") -> wp.array:
        """Reshapes and arranges 3D data to (num_bodies * num_instances, data_dim).

        Our internal methods consume and return data arranged as ``(num_instances, num_bodies, data_dim)``::

            [
                [[env_0_body_0_data_0, env_0_body_0_data_1, ...], [env_0_body_1_data_0, env_0_body_1_data_1, ...], ...],
                [[env_1_body_0_data_0, env_1_body_0_data_1, ...], [env_1_body_1_data_0, env_1_body_1_data_1, ...], ...],
                ...,
            ]

        The view needs data ordered as ``(num_bodies * num_instances, data_dim)``::

            [[body0_env0_data_0, body0_env0_data_1, ...], [body0_env1_data_0, body0_env1_data_1, ...], ...]

        Args:
            data: The data to be formatted for the view. Shape is (num_instances, num_bodies, data_dim).
            data_dim: The trailing dimension size.

        Returns:
            The data formatted for the view. Shape is (num_bodies * num_instances, data_dim).
        """
        element_size = wp.types.type_size_in_bytes(data.dtype)
        row_size = element_size * data_dim
        strided_view = wp.array(
            ptr=data.ptr,
            shape=(self.num_bodies, self.num_instances, data_dim),
            dtype=data.dtype,
            strides=(row_size, self.num_bodies * row_size, element_size),
            device=data.device,
        )
        # Clone to make contiguous (now row-major num_bodies x num_instances x data_dim), then flatten
        return wp.clone(strided_view, device=device).reshape((self.num_bodies * self.num_instances, data_dim))

    """
    Internal helper.
    """

    def _resolve_env_ids(self, env_ids) -> wp.array:
        """Resolve environment indices to a warp array."""
        if isinstance(env_ids, list):
            return wp.array(env_ids, dtype=wp.int32, device=self.device)
        if (env_ids is None) or (env_ids == slice(None)):
            return self._ALL_ENV_INDICES
        if isinstance(env_ids, torch.Tensor):
            return wp.from_torch(env_ids.to(torch.int32), dtype=wp.int32)
        return env_ids

    def _resolve_body_ids(self, body_ids) -> wp.array:
        """Resolve body indices to a warp array."""
        if body_ids is None or (body_ids == slice(None)):
            return self._ALL_BODY_INDICES
        if isinstance(body_ids, slice):
            return wp.from_torch(
                torch.arange(self.num_bodies, dtype=torch.int32, device=self.device)[body_ids], dtype=wp.int32
            )
        if isinstance(body_ids, list):
            return wp.array(body_ids, dtype=wp.int32, device=self.device)
        if isinstance(body_ids, torch.Tensor):
            return wp.from_torch(body_ids.to(torch.int32), dtype=wp.int32)
        return body_ids

    def _initialize_impl(self):
        # clear body names list to prevent double counting on re-initialization
        self._body_names_list.clear()
        # obtain global simulation view
        self._physics_sim_view = SimulationManager.get_physics_sim_view()
        root_prim_path_exprs = []
        for name, rigid_body_cfg in self.cfg.rigid_objects.items():
            # obtain the first prim in the regex expression (all others are assumed to be a copy of this)
            template_prim = sim_utils.find_first_matching_prim(rigid_body_cfg.prim_path)
            if template_prim is None:
                raise RuntimeError(f"Failed to find prim for expression: '{rigid_body_cfg.prim_path}'.")
            template_prim_path = template_prim.GetPath().pathString

            # find rigid root prims
            root_prims = sim_utils.get_all_matching_child_prims(
                template_prim_path,
                predicate=lambda prim: prim.HasAPI(UsdPhysics.RigidBodyAPI),
                traverse_instance_prims=False,
            )
            if len(root_prims) == 0:
                raise RuntimeError(
                    f"Failed to find a rigid body when resolving '{rigid_body_cfg.prim_path}'."
                    " Please ensure that the prim has 'USD RigidBodyAPI' applied."
                )
            if len(root_prims) > 1:
                raise RuntimeError(
                    f"Failed to find a single rigid body when resolving '{rigid_body_cfg.prim_path}'."
                    f" Found multiple '{root_prims}' under '{template_prim_path}'."
                    " Please ensure that there is only one rigid body in the prim path tree."
                )

            # check that no rigid object has an articulation root API, which decreases simulation performance
            articulation_prims = sim_utils.get_all_matching_child_prims(
                template_prim_path,
                predicate=lambda prim: prim.HasAPI(UsdPhysics.ArticulationRootAPI),
                traverse_instance_prims=False,
            )
            if len(articulation_prims) != 0:
                if articulation_prims[0].GetAttribute("physxArticulation:articulationEnabled").Get():
                    raise RuntimeError(
                        f"Found an articulation root when resolving '{rigid_body_cfg.prim_path}' in the rigid object"
                        f" collection. These are located at: '{articulation_prims}' under '{template_prim_path}'."
                        " Please disable the articulation root in the USD or from code by setting the parameter"
                        " 'ArticulationRootPropertiesCfg.articulation_enabled' to False in the spawn configuration."
                    )

            # resolve root prim back into regex expression
            root_prim_path = root_prims[0].GetPath().pathString
            root_prim_path_expr = rigid_body_cfg.prim_path + root_prim_path[len(template_prim_path) :]
            root_prim_path_exprs.append(root_prim_path_expr.replace(".*", "*"))

            self._body_names_list.append(name)

        # -- object view
        self._root_view = self._physics_sim_view.create_rigid_body_view(root_prim_path_exprs)

        # check if the rigid body was created
        if self._root_view._backend is None:
            raise RuntimeError("Failed to create rigid body collection. Please check PhysX logs.")

        # log information about the rigid body
        logger.info(f"Number of instances: {self.num_instances}")
        logger.info(f"Number of distinct bodies: {self.num_bodies}")
        logger.info(f"Body names: {self.body_names}")

        # container for data access
        self._data = RigidObjectCollectionData(self.root_view, self.num_bodies, self.device)

        # create buffers
        self._create_buffers()
        # process configuration
        self._process_cfg()
        # update the rigid body data
        self.update(0.0)

    def _create_buffers(self):
        # constants
        self._ALL_ENV_INDICES = wp.array(
            np.arange(self.num_instances, dtype=np.int32), device=self.device, dtype=wp.int32
        )
        self._ALL_BODY_INDICES = wp.array(
            np.arange(self.num_bodies, dtype=np.int32), device=self.device, dtype=wp.int32
        )

        # external wrench composer
        self._instantaneous_wrench_composer = WrenchComposer(self)
        self._permanent_wrench_composer = WrenchComposer(self)

        # set information about rigid body into data
        self._data.body_names = self.body_names

    def _process_cfg(self) -> None:
        """Post processing of configuration parameters."""
        # default state
        # -- body state
        default_body_poses = []
        default_body_vels = []
        for rigid_object_cfg in self.cfg.rigid_objects.values():
            default_body_pose = tuple(rigid_object_cfg.init_state.pos) + tuple(rigid_object_cfg.init_state.rot)
            default_body_vel = tuple(rigid_object_cfg.init_state.lin_vel) + tuple(rigid_object_cfg.init_state.ang_vel)
            default_body_pose = np.tile(np.array(default_body_pose, dtype=np.float32), (self.num_instances, 1))
            default_body_vel = np.tile(np.array(default_body_vel, dtype=np.float32), (self.num_instances, 1))
            default_body_poses.append(default_body_pose)
            default_body_vels.append(default_body_vel)
        # Stack: each has shape (num_instances, data_size) -> (num_instances, num_bodies, data_size)
        default_body_poses = np.stack(default_body_poses, axis=1)
        default_body_vels = np.stack(default_body_vels, axis=1)
        self.data.default_body_pose = wp.array(default_body_poses, dtype=wp.transformf, device=self.device)
        self.data.default_body_vel = wp.array(default_body_vels, dtype=wp.spatial_vectorf, device=self.device)

    def _env_body_ids_to_view_ids(
        self, env_ids: torch.Tensor | wp.array, body_ids: torch.Tensor | wp.array, device: str = "cuda:0"
    ) -> wp.array:
        """Converts environment and body indices to indices consistent with data from :attr:`root_view`.

        Args:
            env_ids: Environment indices.
            body_ids: Body indices.

        Returns:
            The view indices.
        """
        # the order is body_0/env_0, body_0/env_1, body_0/env_..., body_1/env_0, body_1/env_1, ...
        # return a flat tensor of indices
        num_query_envs = env_ids.shape[0]
        view_ids = wp.zeros(num_query_envs * body_ids.shape[0], dtype=wp.int32, device=device)
        wp.launch(
            resolve_view_ids,
            dim=(num_query_envs, body_ids.shape[0]),
            inputs=[env_ids, body_ids, num_query_envs, self.num_instances],
            outputs=[view_ids],
            device=device,
        )
        return view_ids

    """
    Internal simulation callbacks.
    """

    def _invalidate_initialize_callback(self, event) -> None:
        """Invalidates the scene elements."""
        # call parent
        super()._invalidate_initialize_callback(event)
        # set all existing views to None to invalidate them
        self._root_view = None

    def _on_prim_deletion(self, prim_path: str) -> None:
        """Invalidates and deletes the callbacks when the prim is deleted.

        Args:
            prim_path: The path to the prim that is being deleted.

        .. note::
            This function is called when the prim is deleted.
        """
        if prim_path == "/":
            self._clear_callbacks()
            return
        for prim_path_expr in [obj.prim_path for obj in self.cfg.rigid_objects.values()]:
            result = re.match(
                pattern="^" + "/".join(prim_path_expr.split("/")[: prim_path.count("/") + 1]) + "$", string=prim_path
            )
            if result:
                self._clear_callbacks()
                return

    """
    Deprecated properties and methods.
    """

    @property
    def root_physx_view(self) -> physx.RigidBodyView:
        """Deprecated property. Please use :attr:`root_view` instead."""
        warnings.warn(
            "The `root_physx_view` property will be deprecated in a future release. Please use `root_view` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.root_view

    def write_body_state_to_sim(
        self,
        body_states: torch.Tensor | wp.array,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        body_ids: slice | torch.Tensor | None = None,
    ) -> None:
        """Deprecated, same as :meth:`write_body_link_pose_to_sim_index` and
        :meth:`write_body_com_velocity_to_sim_index`."""
        warnings.warn(
            "The function 'write_body_state_to_sim' will be deprecated in a future release. Please"
            " use 'write_body_link_pose_to_sim_index' and 'write_body_com_velocity_to_sim_index' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.write_body_link_pose_to_sim_index(body_states[:, :, :7], env_ids=env_ids, body_ids=body_ids)
        self.write_body_com_velocity_to_sim_index(body_states[:, :, 7:], env_ids=env_ids, body_ids=body_ids)

    def write_body_com_state_to_sim(
        self,
        body_states: torch.Tensor | wp.array,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        body_ids: slice | torch.Tensor | None = None,
    ) -> None:
        """Deprecated, same as :meth:`write_body_com_pose_to_sim_index` and
        :meth:`write_body_com_velocity_to_sim_index`."""
        warnings.warn(
            "The function 'write_body_com_state_to_sim' will be deprecated in a future release. Please"
            " use 'write_body_com_pose_to_sim_index' and 'write_body_com_velocity_to_sim_index' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.write_body_com_pose_to_sim_index(body_states[:, :, :7], env_ids=env_ids, body_ids=body_ids)
        self.write_body_com_velocity_to_sim_index(body_states[:, :, 7:], env_ids=env_ids, body_ids=body_ids)

    def write_body_link_state_to_sim(
        self,
        body_states: torch.Tensor | wp.array,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        body_ids: slice | torch.Tensor | None = None,
    ) -> None:
        """Deprecated, same as :meth:`write_body_link_pose_to_sim_index` and
        :meth:`write_body_link_velocity_to_sim_index`."""
        warnings.warn(
            "The function 'write_body_link_state_to_sim' will be deprecated in a future release. Please"
            " use 'write_body_link_pose_to_sim_index' and 'write_body_link_velocity_to_sim_index' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.write_body_link_pose_to_sim_index(body_states[:, :, :7], env_ids=env_ids, body_ids=body_ids)
        self.write_body_link_velocity_to_sim_index(body_states[:, :, 7:], env_ids=env_ids, body_ids=body_ids)
