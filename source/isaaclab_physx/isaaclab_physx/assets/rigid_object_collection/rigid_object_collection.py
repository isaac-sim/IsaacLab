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

import torch
import warp as wp

import omni.physics.tensors.impl.api as physx
from pxr import UsdPhysics

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
import isaaclab.utils.string as string_utils
from isaaclab.assets.rigid_object_collection.base_rigid_object_collection import BaseRigidObjectCollection
from isaaclab.utils.wrench_composer import WrenchComposer

from isaaclab_physx.physics import PhysxManager as SimulationManager

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
        self._prim_paths = []
        # spawn the rigid objects
        for rigid_body_cfg in self.cfg.rigid_objects.values():
            # check if the rigid object path is valid
            # note: currently the spawner does not work if there is a regex pattern in the leaf
            #   For example, if the prim path is "/World/Object_[1,2]" since the spawner will not
            #   know which prim to spawn. This is a limitation of the spawner and not the asset.
            asset_path = rigid_body_cfg.prim_path.split("/")[-1]
            asset_path_is_regex = re.match(r"^[a-zA-Z0-9/_]+$", asset_path) is None
            # spawn the asset
            if rigid_body_cfg.spawn is not None and not asset_path_is_regex:
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
            self._prim_paths.append(rigid_body_cfg.prim_path)
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
                    body_ids=self._ALL_BODY_INDICES_WP,
                    env_ids=self._ALL_ENV_INDICES_WP,
                )
                # Apply both instantaneous and permanent wrench to the simulation
                self.root_view.apply_forces_and_torques_at_position(
                    force_data=self.reshape_data_to_view(self._instantaneous_wrench_composer.composed_force_as_torch),
                    torque_data=self.reshape_data_to_view(self._instantaneous_wrench_composer.composed_torque_as_torch),
                    position_data=None,
                    indices=self._env_body_ids_to_view_ids(self._ALL_ENV_INDICES, self._ALL_BODY_INDICES),
                    is_global=False,
                )
            else:
                # Apply permanent wrench to the simulation
                self.root_view.apply_forces_and_torques_at_position(
                    force_data=self.reshape_data_to_view(self._permanent_wrench_composer.composed_force_as_torch),
                    torque_data=self.reshape_data_to_view(self._permanent_wrench_composer.composed_torque_as_torch),
                    position_data=None,
                    indices=self._env_body_ids_to_view_ids(self._ALL_ENV_INDICES, self._ALL_BODY_INDICES),
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
    ) -> tuple[torch.Tensor, list[str], list[int]]:
        """Find bodies in the rigid body collection based on the name keys.

        Please check the :meth:`isaaclab.utils.string_utils.resolve_matching_names` function for more
        information on the name matching.

        Args:
            name_keys: A regular expression or a list of regular expressions to match the body names.
            preserve_order: Whether to preserve the order of the name keys in the output. Defaults to False.

        Returns:
            A tuple of lists containing the body mask, names and indices.
        """
        obj_ids, obj_names = string_utils.resolve_matching_names(name_keys, self.object_names, preserve_order)
        return torch.tensor(obj_ids, device=self.device), obj_names

    """
    Operations - Write to simulation.
    """

    def write_body_state_to_sim(
        self,
        body_states: torch.Tensor,
        env_ids: torch.Tensor | None = None,
        body_ids: slice | torch.Tensor | None = None,
    ) -> None:
        """Set the bodies state over selected environment indices into the simulation.

        The body state comprises of the cartesian position, quaternion orientation in (x, y, z, w), and linear
        and angular velocity. All the quantities are in the simulation frame. Shape is
        (len(env_ids), len(body_ids), 13).

        Args:
            body_states: Body states in simulation frame. Shape is (len(env_ids), len(body_ids), 13).
            env_ids: Environment indices. If None, then all indices are used.
            body_ids: Body indices. If None, then all indices are used.
        """
        self.write_body_link_pose_to_sim(body_states[..., :7], env_ids=env_ids, body_ids=body_ids)
        self.write_body_com_velocity_to_sim(body_states[..., 7:], env_ids=env_ids, body_ids=body_ids)

    def write_body_com_state_to_sim(
        self,
        body_states: torch.Tensor,
        env_ids: torch.Tensor | None = None,
        body_ids: slice | torch.Tensor | None = None,
    ) -> None:
        """Set the body center of mass state over selected environment and body indices into the simulation.

        The body state comprises of the cartesian position, quaternion orientation in (x, y, z, w), and linear
        and angular velocity. All the quantities are in the simulation frame.

        Args:
            body_states: Body states in simulation frame. Shape is (len(env_ids), len(body_ids), 13).
            env_ids: Environment indices. If None, then all indices are used.
            body_ids: Body indices. If None, then all indices are used.
        """
        self.write_body_com_pose_to_sim(body_states[..., :7], env_ids=env_ids, body_ids=body_ids)
        self.write_body_com_velocity_to_sim(body_states[..., 7:], env_ids=env_ids, body_ids=body_ids)

    def write_body_link_state_to_sim(
        self,
        body_states: torch.Tensor,
        env_ids: torch.Tensor | None = None,
        body_ids: slice | torch.Tensor | None = None,
    ) -> None:
        """Set the body link state over selected environment and body indices into the simulation.

        The body state comprises of the cartesian position, quaternion orientation in (x, y, z, w), and linear
        and angular velocity. All the quantities are in the simulation frame.

        Args:
            body_states: Body states in simulation frame. Shape is (len(env_ids), len(body_ids), 13).
            env_ids: Environment indices. If None, then all indices are used.
            body_ids: Body indices. If None, then all indices are used.
        """
        self.write_body_link_pose_to_sim(body_states[..., :7], env_ids=env_ids, body_ids=body_ids)
        self.write_body_link_velocity_to_sim(body_states[..., 7:], env_ids=env_ids, body_ids=body_ids)

    def write_body_pose_to_sim(
        self,
        body_poses: torch.Tensor,
        env_ids: torch.Tensor | None = None,
        body_ids: slice | torch.Tensor | None = None,
    ) -> None:
        """Set the body poses over selected environment and body indices into the simulation.

        The body pose comprises of the cartesian position and quaternion orientation in (x, y, z, w).

        Args:
            body_poses: Body poses in simulation frame. Shape is (len(env_ids), len(body_ids), 7).
            env_ids: Environment indices. If None, then all indices are used.
            body_ids: Body indices. If None, then all indices are used.
        """
        self.write_body_link_pose_to_sim(body_poses, env_ids=env_ids, body_ids=body_ids)

    def write_body_link_pose_to_sim(
        self,
        body_poses: torch.Tensor,
        env_ids: torch.Tensor | None = None,
        body_ids: slice | torch.Tensor | None = None,
    ) -> None:
        """Set the body link pose over selected environment and body indices into the simulation.

        The body link pose comprises of the cartesian position and quaternion orientation in (x, y, z, w).

        Args:
            body_poses: Body link poses in simulation frame. Shape is (len(env_ids), len(body_ids), 7).
            env_ids: Environment indices. If None, then all indices are used.
            body_ids: Body indices. If None, then all indices are used.
        """
        # resolve all indices
        # -- env_ids
        if env_ids is None:
            env_ids = self._ALL_ENV_INDICES
        # -- body_ids
        if body_ids is None:
            body_ids = self._ALL_BODY_INDICES

        # convert lists to tensors for proper indexing
        if isinstance(env_ids, list):
            env_ids = torch.tensor(env_ids, dtype=torch.long, device=self.device)
        if isinstance(body_ids, list):
            body_ids = torch.tensor(body_ids, dtype=torch.long, device=self.device)

        # note: we need to do this here since tensors are not set into simulation until step.
        # set into internal buffers
        self.data.body_link_pose_w[env_ids[:, None], body_ids] = body_poses.clone()
        # update these buffers only if the user is using them. Otherwise this adds to overhead.
        if self.data._body_link_state_w.data is not None:
            self.data.body_link_state_w[env_ids[:, None], body_ids, :7] = body_poses.clone()
        if self.data._body_state_w.data is not None:
            self.data.body_state_w[env_ids[:, None], body_ids, :7] = body_poses.clone()
        if self.data._body_com_state_w.data is not None:
            # get CoM pose in link frame
            com_pos_b = self.data.body_com_pos_b[env_ids[:, None], body_ids]
            com_quat_b = self.data.body_com_quat_b[env_ids[:, None], body_ids]
            com_pos, com_quat = math_utils.combine_frame_transforms(
                body_poses[..., :3],
                body_poses[..., 3:7],
                com_pos_b,
                com_quat_b,
            )
            self.data.body_com_state_w[env_ids[:, None], body_ids, :3] = com_pos
            self.data.body_com_state_w[env_ids[:, None], body_ids, 3:7] = com_quat

        poses_xyzw = self.data.body_link_pose_w.clone()

        # set into simulation
        view_ids = self._env_body_ids_to_view_ids(env_ids, body_ids)
        self.root_view.set_transforms(self.reshape_data_to_view(poses_xyzw), indices=view_ids)

    def write_body_com_pose_to_sim(
        self,
        body_poses: torch.Tensor,
        env_ids: torch.Tensor | None = None,
        body_ids: slice | torch.Tensor | None = None,
    ) -> None:
        """Set the body center of mass pose over selected environment and body indices into the simulation.

        The body center of mass pose comprises of the cartesian position and quaternion orientation in (x, y, z, w).
        The orientation is the orientation of the principle axes of inertia.

        Args:
            body_poses: Body center of mass poses in simulation frame. Shape is (len(env_ids), len(body_ids), 7).
            env_ids: Environment indices. If None, then all indices are used.
            body_ids: Body indices. If None, then all indices are used.
        """
        # resolve all indices
        # -- env_ids
        if env_ids is None:
            env_ids = self._ALL_ENV_INDICES
        # -- body_ids
        if body_ids is None:
            body_ids = self._ALL_BODY_INDICES

        # convert lists to tensors for proper indexing
        if isinstance(env_ids, list):
            env_ids = torch.tensor(env_ids, dtype=torch.long, device=self.device)
        if isinstance(body_ids, list):
            body_ids = torch.tensor(body_ids, dtype=torch.long, device=self.device)

        # set into internal buffers
        self.data.body_com_pose_w[env_ids[:, None], body_ids] = body_poses.clone()
        # update these buffers only if the user is using them. Otherwise this adds to overhead.
        if self.data._body_com_state_w.data is not None:
            self.data.body_com_state_w[env_ids[:, None], body_ids, :7] = body_poses.clone()

        # get CoM pose in link frame
        com_pos_b = self.data.body_com_pos_b[env_ids[:, None], body_ids]
        com_quat_b = self.data.body_com_quat_b[env_ids[:, None], body_ids]
        # transform input CoM pose to link frame
        body_link_pos, body_link_quat = math_utils.combine_frame_transforms(
            body_poses[..., :3],
            body_poses[..., 3:7],
            math_utils.quat_apply(math_utils.quat_inv(com_quat_b), -com_pos_b),
            math_utils.quat_inv(com_quat_b),
        )

        # write transformed pose in link frame to sim
        body_link_pose = torch.cat((body_link_pos, body_link_quat), dim=-1)
        self.write_body_link_pose_to_sim(body_link_pose, env_ids=env_ids, body_ids=body_ids)

    def write_body_velocity_to_sim(
        self,
        body_velocities: torch.Tensor,
        env_ids: torch.Tensor | None = None,
        body_ids: slice | torch.Tensor | None = None,
    ) -> None:
        """Set the body velocity over selected environment and body indices into the simulation.

        The velocity comprises linear velocity (x, y, z) and angular velocity (x, y, z) in that order.
        ..note:: This sets the velocity of the body's center of mass rather than the body's frame.

        Args:
            body_velocities: Body velocities in simulation frame. Shape is (len(env_ids), len(body_ids), 6).
            env_ids: Environment indices. If None, then all indices are used.
            body_ids: Body indices. If None, then all indices are used.
        """
        self.write_body_com_velocity_to_sim(body_velocities, env_ids=env_ids, body_ids=body_ids)

    def write_body_com_velocity_to_sim(
        self,
        body_velocities: torch.Tensor,
        env_ids: torch.Tensor | None = None,
        body_ids: slice | torch.Tensor | None = None,
    ) -> None:
        """Set the body center of mass velocity over selected environment and body indices into the simulation.

        The velocity comprises linear velocity (x, y, z) and angular velocity (x, y, z) in that order.
        ..note:: This sets the velocity of the body's center of mass rather than the body's frame.

        Args:
            body_velocities: Body center of mass velocities in simulation frame. Shape is
                (len(env_ids), len(body_ids), 6).
            env_ids: Environment indices. If None, then all indices are used.
            body_ids: Body indices. If None, then all indices are used.
        """
        # resolve all indices
        # -- env_ids
        if env_ids is None:
            env_ids = self._ALL_ENV_INDICES
        # -- body_ids
        if body_ids is None:
            body_ids = self._ALL_BODY_INDICES

        # convert lists to tensors for proper indexing
        if isinstance(env_ids, list):
            env_ids = torch.tensor(env_ids, dtype=torch.long, device=self.device)
        if isinstance(body_ids, list):
            body_ids = torch.tensor(body_ids, dtype=torch.long, device=self.device)

        # note: we need to do this here since tensors are not set into simulation until step.
        # set into internal buffers
        self.data.body_com_vel_w[env_ids[:, None], body_ids] = body_velocities.clone()
        # update these buffers only if the user is using them. Otherwise this adds to overhead.
        if self.data._body_com_state_w.data is not None:
            self.data.body_com_state_w[env_ids[:, None], body_ids, 7:] = body_velocities.clone()
        if self.data._body_state_w.data is not None:
            self.data.body_state_w[env_ids[:, None], body_ids, 7:] = body_velocities.clone()
        if self.data._body_link_state_w.data is not None:
            self.data.body_link_state_w[env_ids[:, None], body_ids, 7:] = body_velocities.clone()
        # make the acceleration zero to prevent reporting old values
        self.data.body_com_acc_w[env_ids[:, None], body_ids] = 0.0

        # set into simulation
        view_ids = self._env_body_ids_to_view_ids(env_ids, body_ids)
        self.root_view.set_velocities(self.reshape_data_to_view(self.data.body_com_vel_w), indices=view_ids)

    def write_body_link_velocity_to_sim(
        self,
        body_velocities: torch.Tensor,
        env_ids: torch.Tensor | None = None,
        body_ids: slice | torch.Tensor | None = None,
    ) -> None:
        """Set the body link velocity over selected environment and body indices into the simulation.

        The velocity comprises linear velocity (x, y, z) and angular velocity (x, y, z) in that order.
        ..note:: This sets the velocity of the body's frame rather than the body's center of mass.

        Args:
            body_velocities: Body link velocities in simulation frame. Shape is (len(env_ids), len(body_ids), 6).
            env_ids: Environment indices. If None, then all indices are used.
            body_ids: Body indices. If None, then all indices are used.
        """
        # resolve all indices
        # -- env_ids
        if env_ids is None:
            env_ids = self._ALL_ENV_INDICES
        # -- body_ids
        if body_ids is None:
            body_ids = self._ALL_BODY_INDICES

        # convert lists to tensors for proper indexing
        if isinstance(env_ids, list):
            env_ids = torch.tensor(env_ids, dtype=torch.long, device=self.device)
        if isinstance(body_ids, list):
            body_ids = torch.tensor(body_ids, dtype=torch.long, device=self.device)

        # set into internal buffers
        self.data.body_link_vel_w[env_ids[:, None], body_ids] = body_velocities.clone()
        # update these buffers only if the user is using them. Otherwise this adds to overhead.
        if self.data._body_link_state_w.data is not None:
            self.data.body_link_state_w[env_ids[:, None], body_ids, 7:] = body_velocities.clone()

        # get CoM pose in link frame
        quat = self.data.body_link_quat_w[env_ids[:, None], body_ids]
        com_pos_b = self.data.body_com_pos_b[env_ids[:, None], body_ids]
        # transform input velocity to center of mass frame
        body_com_velocity = body_velocities.clone()
        body_com_velocity[..., :3] += torch.linalg.cross(
            body_com_velocity[..., 3:], math_utils.quat_apply(quat, com_pos_b), dim=-1
        )

        # write center of mass velocity to sim
        self.write_body_com_velocity_to_sim(body_com_velocity, env_ids=env_ids, body_ids=body_ids)

    """
    Operations - Setters.
    """

    def set_masses(
        self,
        masses: torch.Tensor,
        body_ids: torch.Tensor | None = None,
        env_ids: torch.Tensor | None = None,
    ):
        """Set masses of all bodies.

        Args:
            masses: Masses of all bodies. Shape is (len(env_ids), len(body_ids)).
            body_ids: The body indices to set the masses for. Defaults to None (all bodies).
            env_ids: The environment indices to set the masses for. Defaults to None (all environments).
        """
        # resolve indices
        if env_ids is None:
            env_ids = self._ALL_ENV_INDICES
        if body_ids is None:
            body_ids = self._ALL_BODY_INDICES

        # convert lists to tensors for proper indexing
        if isinstance(env_ids, list):
            env_ids = torch.tensor(env_ids, dtype=torch.long, device=self.device)
        if isinstance(body_ids, list):
            body_ids = torch.tensor(body_ids, dtype=torch.long, device=self.device)

        # set into internal buffers
        # _body_mass shape from view is (num_instances * num_bodies, 1)
        # We need to update only the selected env_ids and body_ids
        view_ids = self._env_body_ids_to_view_ids(env_ids, body_ids)
        # masses input shape is (len(env_ids), len(body_ids)), flatten to match view
        self.data._body_mass[view_ids] = masses.reshape(-1, 1)

        # set into simulation
        self.root_view.set_masses(self.data._body_mass.cpu(), indices=view_ids.cpu())

    def set_coms(
        self,
        coms: torch.Tensor,
        body_ids: torch.Tensor | None = None,
        env_ids: torch.Tensor | None = None,
    ):
        """Set center of mass positions of all bodies.

        Args:
            coms: Center of mass positions of all bodies. Shape is (len(env_ids), len(body_ids), 3).
            body_ids: The body indices to set the center of mass positions for. Defaults to None (all bodies).
            env_ids: The environment indices to set the center of mass positions for. Defaults to None
                (all environments).
        """
        # resolve indices
        if env_ids is None:
            env_ids = self._ALL_ENV_INDICES
        if body_ids is None:
            body_ids = self._ALL_BODY_INDICES

        # convert lists to tensors for proper indexing
        if isinstance(env_ids, list):
            env_ids = torch.tensor(env_ids, dtype=torch.long, device=self.device)
        if isinstance(body_ids, list):
            body_ids = torch.tensor(body_ids, dtype=torch.long, device=self.device)

        # get view indices
        view_ids = self._env_body_ids_to_view_ids(env_ids, body_ids)

        # get current com poses and update position part
        # body_com_pose_b triggers lazy evaluation, so we work with the underlying buffer
        current_poses = self.root_view.get_coms().to(self.device)
        # coms input shape is (len(env_ids), len(body_ids), 3), flatten to (N, 3)
        current_poses[view_ids, :3] = coms.reshape(-1, 3)

        # set into simulation
        self.root_view.set_coms(current_poses.cpu(), indices=view_ids.cpu())

        # invalidate the cached buffer
        self.data._body_com_pose_b.timestamp = -1

    def set_inertias(
        self,
        inertias: torch.Tensor,
        body_ids: torch.Tensor | None = None,
        env_ids: torch.Tensor | None = None,
    ):
        """Set inertias of all bodies.

        Args:
            inertias: Inertias of all bodies. Shape is (len(env_ids), len(body_ids), 3, 3).
            body_ids: The body indices to set the inertias for. Defaults to None (all bodies).
            env_ids: The environment indices to set the inertias for. Defaults to None (all environments).
        """
        # resolve indices
        if env_ids is None:
            env_ids = self._ALL_ENV_INDICES
        if body_ids is None:
            body_ids = self._ALL_BODY_INDICES

        # convert lists to tensors for proper indexing
        if isinstance(env_ids, list):
            env_ids = torch.tensor(env_ids, dtype=torch.long, device=self.device)
        if isinstance(body_ids, list):
            body_ids = torch.tensor(body_ids, dtype=torch.long, device=self.device)

        # get view indices
        view_ids = self._env_body_ids_to_view_ids(env_ids, body_ids)

        # set into internal buffers
        # _body_inertia shape from view is (num_instances * num_bodies, 9) - flattened 3x3 matrices
        # inertias input shape is (len(env_ids), len(body_ids), 3, 3), flatten to (N, 9)
        self.data._body_inertia[view_ids] = inertias.reshape(-1, 9)

        # set into simulation
        self.root_view.set_inertias(self.data._body_inertia.cpu(), indices=view_ids.cpu())

    def set_external_force_and_torque(
        self,
        forces: torch.Tensor,
        torques: torch.Tensor,
        positions: torch.Tensor | None = None,
        body_ids: torch.Tensor | slice | None = None,
        env_ids: torch.Tensor | None = None,
        is_global: bool = False,
    ) -> None:
        """Set external force and torque to apply on the rigid object collection's bodies in their local frame.

        For many applications, we want to keep the applied external force on rigid bodies constant over a period of
        time (for instance, during the policy control). This function allows us to store the external force and torque
        into buffers which are then applied to the simulation at every step. Optionally, set the position to apply the
        external wrench at (in the local link frame of the bodies).

        .. caution::
            If the function is called with empty forces and torques, then this function disables the application
            of external wrench to the simulation.

            .. code-block:: python

                # example of disabling external wrench
                asset.set_external_force_and_torque(forces=torch.zeros(0, 3), torques=torch.zeros(0, 3))

        .. caution::
            If the function is called consecutively with and with different values for ``is_global``, then the
            all the external wrenches will be applied in the frame specified by the last call.

            .. code-block:: python

                # example of setting external wrench in the global frame
                asset.set_external_force_and_torque(forces=torch.ones(1, 1, 3), env_ids=[0], is_global=True)
                # example of setting external wrench in the link frame
                asset.set_external_force_and_torque(forces=torch.ones(1, 1, 3), env_ids=[1], is_global=False)
                # Both environments will have the external wrenches applied in the link frame

        .. note::
            This function does not apply the external wrench to the simulation. It only fills the buffers with
            the desired values. To apply the external wrench, call the :meth:`write_data_to_sim` function
            right before the simulation step.

        Args:
            forces: External forces in bodies' local frame. Shape is (len(env_ids), len(body_ids), 3).
            torques: External torques in bodies' local frame. Shape is (len(env_ids), len(body_ids), 3).
            positions: External wrench positions in bodies' local frame. Shape is (len(env_ids), len(body_ids), 3).
                Defaults to None.
            body_ids: Body indices to apply external wrench to. Defaults to None (all bodies).
            env_ids: Environment indices to apply external wrench to. Defaults to None (all instances).
            is_global: Whether to apply the external wrench in the global frame. Defaults to False. If set to False,
                the external wrench is applied in the link frame of the bodies.
        """
        logger.warning(
            "The function 'set_external_force_and_torque' will be deprecated in a future release. Please"
            " use 'permanent_wrench_composer.set_forces_and_torques' instead."
        )

        if forces is None and torques is None:
            logger.warning("No forces or torques provided. No permanent external wrench will be applied.")

        # resolve all indices
        # -- env_ids
        if env_ids is None:
            env_ids = self._ALL_ENV_INDICES_WP
        elif not isinstance(env_ids, torch.Tensor):
            env_ids = wp.array(env_ids, dtype=wp.int32, device=self.device)
        else:
            env_ids = wp.from_torch(env_ids.to(torch.int32), dtype=wp.int32)
        # -- body_ids
        if body_ids is None:
            body_ids = self._ALL_BODY_INDICES_WP
        elif isinstance(body_ids, slice):
            body_ids = wp.from_torch(
                torch.arange(self.num_bodies, dtype=torch.int32, device=self.device)[body_ids], dtype=wp.int32
            )
        elif not isinstance(body_ids, torch.Tensor):
            body_ids = wp.array(body_ids, dtype=wp.int32, device=self.device)
        else:
            body_ids = wp.from_torch(body_ids.to(torch.int32), dtype=wp.int32)

        # Write to wrench composer
        self._permanent_wrench_composer.set_forces_and_torques(
            forces=wp.from_torch(forces, dtype=wp.vec3f) if forces is not None else None,
            torques=wp.from_torch(torques, dtype=wp.vec3f) if torques is not None else None,
            positions=wp.from_torch(positions, dtype=wp.vec3f) if positions is not None else None,
            body_ids=body_ids,
            env_ids=env_ids,
            is_global=is_global,
        )

    """
    Helper functions.
    """

    def reshape_view_to_data(self, data: torch.Tensor) -> torch.Tensor:
        """Reshapes and arranges the data coming from the :attr:`root_view` to
        (num_instances, num_bodies, data_dim).

        Args:
            data: The data coming from the :attr:`root_view`. Shape is (num_instances * num_bodies, data_dim).

        Returns:
            The reshaped data. Shape is (num_instances, num_bodies, data_dim).
        """
        return torch.einsum("ijk -> jik", data.reshape(self.num_bodies, self.num_instances, -1))

    def reshape_data_to_view(self, data: torch.Tensor) -> torch.Tensor:
        """Reshapes and arranges the data to the be consistent with data from the :attr:`root_view`.

        Args:
            data: The data to be reshaped. Shape is (num_instances, num_bodies, data_dim).

        Returns:
            The reshaped data. Shape is (num_instances * num_bodies, data_dim).
        """
        return torch.einsum("ijk -> jik", data).reshape(self.num_bodies * self.num_instances, *data.shape[2:])

    """
    Internal helper.
    """

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
        self._ALL_ENV_INDICES = torch.arange(self.num_instances, dtype=torch.long, device=self.device)
        self._ALL_BODY_INDICES = torch.arange(self.num_bodies, dtype=torch.long, device=self.device)
        self._ALL_ENV_INDICES_WP = wp.from_torch(self._ALL_ENV_INDICES.to(torch.int32), dtype=wp.int32)
        self._ALL_BODY_INDICES_WP = wp.from_torch(self._ALL_BODY_INDICES.to(torch.int32), dtype=wp.int32)

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
            default_body_pose = (
                torch.tensor(default_body_pose, dtype=torch.float, device=self.device)
                .repeat(self.num_instances, 1)
                .unsqueeze(1)
            )
            default_body_vel = (
                torch.tensor(default_body_vel, dtype=torch.float, device=self.device)
                .repeat(self.num_instances, 1)
                .unsqueeze(1)
            )
            default_body_poses.append(default_body_pose)
            default_body_vels.append(default_body_vel)
        # concatenate the default state for each object
        default_body_poses = torch.cat(default_body_poses, dim=1)
        default_body_vels = torch.cat(default_body_vels, dim=1)
        self.data.default_body_pose = default_body_poses
        self.data.default_body_vel = default_body_vels

    def _env_body_ids_to_view_ids(
        self, env_ids: torch.Tensor, body_ids: torch.Tensor | slice | torch.Tensor
    ) -> torch.Tensor:
        """Converts environment and body indices to indices consistent with data from :attr:`root_view`.

        Args:
            env_ids: Environment indices.
            body_ids: Body indices.

        Returns:
            The view indices.
        """
        # the order is env_0/body_0, env_0/body_1, env_0/body_..., env_1/body_0, env_1/body_1, ...
        # return a flat tensor of indices
        if isinstance(body_ids, slice):
            body_ids = self._ALL_BODY_INDICES
        elif isinstance(body_ids, Sequence):
            body_ids = torch.tensor(body_ids, device=self.device)
        return (body_ids.unsqueeze(1) * self.num_instances + env_ids).flatten()

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
        for prim_path_expr in self._prim_paths:
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
