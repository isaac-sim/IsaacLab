# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import re
import torch
import weakref
from collections.abc import Sequence
from typing import TYPE_CHECKING

import omni.kit.app
import omni.log
import omni.physics.tensors.impl.api as physx
import omni.timeline
from isaacsim.core.simulation_manager import IsaacEvents, SimulationManager
from pxr import UsdPhysics

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
import isaaclab.utils.string as string_utils

from ..asset_base import AssetBase
from .rigid_object_collection_data import RigidObjectCollectionData

if TYPE_CHECKING:
    from .rigid_object_collection_cfg import RigidObjectCollectionCfg


class RigidObjectCollection(AssetBase):
    """A rigid object collection class.

    This class represents a collection of rigid objects in the simulation, where the state of the
    rigid objects can be accessed and modified using a batched ``(env_ids, object_ids)`` API.

    For each rigid body in the collection, the root prim of the asset must have the `USD RigidBodyAPI`_
    applied to it. This API is used to define the simulation properties of the rigid bodies. On playing the
    simulation, the physics engine will automatically register the rigid bodies and create a corresponding
    rigid body handle. This handle can be accessed using the :attr:`root_physx_view` attribute.

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
    """Configuration instance for the rigid object collection."""

    def __init__(self, cfg: RigidObjectCollectionCfg):
        """Initialize the rigid object collection.

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
        for rigid_object_cfg in self.cfg.rigid_objects.values():
            # check if the rigid object path is valid
            # note: currently the spawner does not work if there is a regex pattern in the leaf
            #   For example, if the prim path is "/World/Object_[1,2]" since the spawner will not
            #   know which prim to spawn. This is a limitation of the spawner and not the asset.
            asset_path = rigid_object_cfg.prim_path.split("/")[-1]
            asset_path_is_regex = re.match(r"^[a-zA-Z0-9/_]+$", asset_path) is None
            # spawn the asset
            if rigid_object_cfg.spawn is not None and not asset_path_is_regex:
                rigid_object_cfg.spawn.func(
                    rigid_object_cfg.prim_path,
                    rigid_object_cfg.spawn,
                    translation=rigid_object_cfg.init_state.pos,
                    orientation=rigid_object_cfg.init_state.rot,
                )
            # check that spawn was successful
            matching_prims = sim_utils.find_matching_prims(rigid_object_cfg.prim_path)
            if len(matching_prims) == 0:
                raise RuntimeError(f"Could not find prim with path {rigid_object_cfg.prim_path}.")
            self._prim_paths.append(rigid_object_cfg.prim_path)
        # stores object names
        self._object_names_list = []

        # note: Use weakref on all callbacks to ensure that this object can be deleted when its destructor is called.
        # add callbacks for stage play/stop
        # The order is set to 10 which is arbitrary but should be lower priority than the default order of 0
        timeline_event_stream = omni.timeline.get_timeline_interface().get_timeline_event_stream()
        self._initialize_handle = timeline_event_stream.create_subscription_to_pop_by_type(
            int(omni.timeline.TimelineEventType.PLAY),
            lambda event, obj=weakref.proxy(self): obj._initialize_callback(event),
            order=10,
        )
        self._invalidate_initialize_handle = timeline_event_stream.create_subscription_to_pop_by_type(
            int(omni.timeline.TimelineEventType.STOP),
            lambda event, obj=weakref.proxy(self): obj._invalidate_initialize_callback(event),
            order=10,
        )
        self._prim_deletion_callback_id = SimulationManager.register_callback(
            self._on_prim_deletion, event=IsaacEvents.PRIM_DELETION
        )
        self._debug_vis_handle = None

    """
    Properties
    """

    @property
    def data(self) -> RigidObjectCollectionData:
        return self._data

    @property
    def num_instances(self) -> int:
        """Number of instances of the collection."""
        return self.root_physx_view.count // self.num_objects

    @property
    def num_objects(self) -> int:
        """Number of objects in the collection.

        This corresponds to the distinct number of rigid bodies in the collection.
        """
        return len(self.object_names)

    @property
    def object_names(self) -> list[str]:
        """Ordered names of objects in the rigid object collection."""
        return self._object_names_list

    @property
    def root_physx_view(self) -> physx.RigidBodyView:
        """Rigid body view for the rigid body collection (PhysX).

        Note:
            Use this view with caution. It requires handling of tensors in a specific way.
        """
        return self._root_physx_view  # type: ignore

    """
    Operations.
    """

    def reset(self, env_ids: torch.Tensor | None = None, object_ids: slice | torch.Tensor | None = None):
        """Resets all internal buffers of selected environments and objects.

        Args:
            env_ids: The indices of the object to reset. Defaults to None (all instances).
            object_ids: The indices of the object to reset. Defaults to None (all objects).
        """
        # resolve all indices
        if env_ids is None:
            env_ids = self._ALL_ENV_INDICES
        if object_ids is None:
            object_ids = self._ALL_OBJ_INDICES
        # reset external wrench
        self._external_force_b[env_ids[:, None], object_ids] = 0.0
        self._external_torque_b[env_ids[:, None], object_ids] = 0.0

    def write_data_to_sim(self):
        """Write external wrench to the simulation.

        Note:
            We write external wrench to the simulation here since this function is called before the simulation step.
            This ensures that the external wrench is applied at every simulation step.
        """
        # write external wrench
        if self.has_external_wrench:
            self.root_physx_view.apply_forces_and_torques_at_position(
                force_data=self.reshape_data_to_view(self._external_force_b),
                torque_data=self.reshape_data_to_view(self._external_torque_b),
                position_data=None,
                indices=self._env_obj_ids_to_view_ids(self._ALL_ENV_INDICES, self._ALL_OBJ_INDICES),
                is_global=False,
            )

    def update(self, dt: float):
        self._data.update(dt)

    """
    Operations - Finders.
    """

    def find_objects(
        self, name_keys: str | Sequence[str], preserve_order: bool = False
    ) -> tuple[torch.Tensor, list[str]]:
        """Find objects in the collection based on the name keys.

        Please check the :meth:`isaaclab.utils.string_utils.resolve_matching_names` function for more
        information on the name matching.

        Args:
            name_keys: A regular expression or a list of regular expressions to match the object names.
            preserve_order: Whether to preserve the order of the name keys in the output. Defaults to False.

        Returns:
            A tuple containing the object indices and names.
        """
        obj_ids, obj_names = string_utils.resolve_matching_names(name_keys, self.object_names, preserve_order)
        return torch.tensor(obj_ids, device=self.device), obj_names

    """
    Operations - Write to simulation.
    """

    def write_object_state_to_sim(
        self,
        object_state: torch.Tensor,
        env_ids: torch.Tensor | None = None,
        object_ids: slice | torch.Tensor | None = None,
    ):
        """Set the object state over selected environment and object indices into the simulation.

        The object state comprises of the cartesian position, quaternion orientation in (w, x, y, z), and linear
        and angular velocity. All the quantities are in the simulation frame.

        Args:
            object_state: Object state in simulation frame. Shape is (len(env_ids), len(object_ids), 13).
            env_ids: Environment indices. If None, then all indices are used.
            object_ids: Object indices. If None, then all indices are used.
        """
        self.write_object_link_pose_to_sim(object_state[..., :7], env_ids=env_ids, object_ids=object_ids)
        self.write_object_com_velocity_to_sim(object_state[..., 7:], env_ids=env_ids, object_ids=object_ids)

    def write_object_com_state_to_sim(
        self,
        object_state: torch.Tensor,
        env_ids: torch.Tensor | None = None,
        object_ids: slice | torch.Tensor | None = None,
    ):
        """Set the object center of mass state over selected environment indices into the simulation.

        The object state comprises of the cartesian position, quaternion orientation in (w, x, y, z), and linear
        and angular velocity. All the quantities are in the simulation frame.

        Args:
            object_state: Object state in simulation frame. Shape is (len(env_ids), len(object_ids), 13).
            env_ids: Environment indices. If None, then all indices are used.
            object_ids: Object indices. If None, then all indices are used.
        """
        self.write_object_com_pose_to_sim(object_state[..., :7], env_ids=env_ids, object_ids=object_ids)
        self.write_object_com_velocity_to_sim(object_state[..., 7:], env_ids=env_ids, object_ids=object_ids)

    def write_object_link_state_to_sim(
        self,
        object_state: torch.Tensor,
        env_ids: torch.Tensor | None = None,
        object_ids: slice | torch.Tensor | None = None,
    ):
        """Set the object link state over selected environment indices into the simulation.

        The object state comprises of the cartesian position, quaternion orientation in (w, x, y, z), and linear
        and angular velocity. All the quantities are in the simulation frame.

        Args:
            object_state: Object state in simulation frame. Shape is (len(env_ids), len(object_ids), 13).
            env_ids: Environment indices. If None, then all indices are used.
            object_ids: Object indices. If None, then all indices are used.
        """
        self.write_object_link_pose_to_sim(object_state[..., :7], env_ids=env_ids, object_ids=object_ids)
        self.write_object_link_velocity_to_sim(object_state[..., 7:], env_ids=env_ids, object_ids=object_ids)

    def write_object_pose_to_sim(
        self,
        object_pose: torch.Tensor,
        env_ids: torch.Tensor | None = None,
        object_ids: slice | torch.Tensor | None = None,
    ):
        """Set the object pose over selected environment and object indices into the simulation.

        The object pose comprises of the cartesian position and quaternion orientation in (w, x, y, z).

        Args:
            object_pose: Object poses in simulation frame. Shape is (len(env_ids), len(object_ids), 7).
            env_ids: Environment indices. If None, then all indices are used.
            object_ids: Object indices. If None, then all indices are used.
        """
        self.write_object_link_pose_to_sim(object_pose, env_ids=env_ids, object_ids=object_ids)

    def write_object_link_pose_to_sim(
        self,
        object_pose: torch.Tensor,
        env_ids: torch.Tensor | None = None,
        object_ids: slice | torch.Tensor | None = None,
    ):
        """Set the object pose over selected environment and object indices into the simulation.

        The object pose comprises of the cartesian position and quaternion orientation in (w, x, y, z).

        Args:
            object_pose: Object poses in simulation frame. Shape is (len(env_ids), len(object_ids), 7).
            env_ids: Environment indices. If None, then all indices are used.
            object_ids: Object indices. If None, then all indices are used.
        """
        # resolve all indices
        # -- env_ids
        if env_ids is None:
            env_ids = self._ALL_ENV_INDICES
        # -- object_ids
        if object_ids is None:
            object_ids = self._ALL_OBJ_INDICES

        # note: we need to do this here since tensors are not set into simulation until step.
        # set into internal buffers
        self._data.object_link_pose_w[env_ids[:, None], object_ids] = object_pose.clone()
        # update these buffers only if the user is using them. Otherwise this adds to overhead.
        if self._data._object_link_state_w.data is not None:
            self._data.object_link_state_w[env_ids[:, None], object_ids, :7] = object_pose.clone()
        if self._data._object_state_w.data is not None:
            self._data.object_state_w[env_ids[:, None], object_ids, :7] = object_pose.clone()
        if self._data._object_com_state_w.data is not None:
            # get CoM pose in link frame
            com_pos_b = self.data.object_com_pos_b[env_ids[:, None], object_ids]
            com_quat_b = self.data.object_com_quat_b[env_ids[:, None], object_ids]
            com_pos, com_quat = math_utils.combine_frame_transforms(
                object_pose[..., :3],
                object_pose[..., 3:7],
                com_pos_b,
                com_quat_b,
            )
            self._data.object_com_state_w[env_ids[:, None], object_ids, :3] = com_pos
            self._data.object_com_state_w[env_ids[:, None], object_ids, 3:7] = com_quat

        # convert the quaternion from wxyz to xyzw
        poses_xyzw = self._data.object_link_pose_w.clone()
        poses_xyzw[..., 3:] = math_utils.convert_quat(poses_xyzw[..., 3:], to="xyzw")

        # set into simulation
        view_ids = self._env_obj_ids_to_view_ids(env_ids, object_ids)
        self.root_physx_view.set_transforms(self.reshape_data_to_view(poses_xyzw), indices=view_ids)

    def write_object_com_pose_to_sim(
        self,
        object_pose: torch.Tensor,
        env_ids: torch.Tensor | None = None,
        object_ids: slice | torch.Tensor | None = None,
    ):
        """Set the object center of mass pose over selected environment indices into the simulation.

        The object pose comprises of the cartesian position and quaternion orientation in (w, x, y, z).
        The orientation is the orientation of the principle axes of inertia.

        Args:
            object_pose: Object poses in simulation frame. Shape is (len(env_ids), len(object_ids), 7).
            env_ids: Environment indices. If None, then all indices are used.
            object_ids: Object indices. If None, then all indices are used.
        """
        # resolve all indices
        # -- env_ids
        if env_ids is None:
            env_ids = self._ALL_ENV_INDICES
        # -- object_ids
        if object_ids is None:
            object_ids = self._ALL_OBJ_INDICES

        # set into internal buffers
        self._data.object_com_pose_w[env_ids[:, None], object_ids] = object_pose.clone()
        # update these buffers only if the user is using them. Otherwise this adds to overhead.
        if self._data._object_com_state_w.data is not None:
            self._data.object_com_state_w[env_ids[:, None], object_ids, :7] = object_pose.clone()

        # get CoM pose in link frame
        com_pos_b = self.data.object_com_pos_b[env_ids[:, None], object_ids]
        com_quat_b = self.data.object_com_quat_b[env_ids[:, None], object_ids]
        # transform input CoM pose to link frame
        object_link_pos, object_link_quat = math_utils.combine_frame_transforms(
            object_pose[..., :3],
            object_pose[..., 3:7],
            math_utils.quat_apply(math_utils.quat_inv(com_quat_b), -com_pos_b),
            math_utils.quat_inv(com_quat_b),
        )

        # write transformed pose in link frame to sim
        object_link_pose = torch.cat((object_link_pos, object_link_quat), dim=-1)
        self.write_object_link_pose_to_sim(object_link_pose, env_ids=env_ids, object_ids=object_ids)

    def write_object_velocity_to_sim(
        self,
        object_velocity: torch.Tensor,
        env_ids: torch.Tensor | None = None,
        object_ids: slice | torch.Tensor | None = None,
    ):
        """Set the object velocity over selected environment and object indices into the simulation.

        Args:
            object_velocity: Object velocities in simulation frame. Shape is (len(env_ids), len(object_ids), 6).
            env_ids: Environment indices. If None, then all indices are used.
            object_ids: Object indices. If None, then all indices are used.
        """
        self.write_object_com_velocity_to_sim(object_velocity, env_ids=env_ids, object_ids=object_ids)

    def write_object_com_velocity_to_sim(
        self,
        object_velocity: torch.Tensor,
        env_ids: torch.Tensor | None = None,
        object_ids: slice | torch.Tensor | None = None,
    ):
        """Set the object center of mass velocity over selected environment and object indices into the simulation.

        Args:
            object_velocity: Object velocities in simulation frame. Shape is (len(env_ids), len(object_ids), 6).
            env_ids: Environment indices. If None, then all indices are used.
            object_ids: Object indices. If None, then all indices are used.
        """
        # resolve all indices
        # -- env_ids
        if env_ids is None:
            env_ids = self._ALL_ENV_INDICES
        # -- object_ids
        if object_ids is None:
            object_ids = self._ALL_OBJ_INDICES

        # note: we need to do this here since tensors are not set into simulation until step.
        # set into internal buffers
        self._data.object_com_vel_w[env_ids[:, None], object_ids] = object_velocity.clone()
        # update these buffers only if the user is using them. Otherwise this adds to overhead.
        if self._data._object_com_state_w.data is not None:
            self._data.object_com_state_w[env_ids[:, None], object_ids, 7:] = object_velocity.clone()
        if self._data._object_state_w.data is not None:
            self._data.object_state_w[env_ids[:, None], object_ids, 7:] = object_velocity.clone()
        if self._data._object_link_state_w.data is not None:
            self._data.object_link_state_w[env_ids[:, None], object_ids, 7:] = object_velocity.clone()
        # make the acceleration zero to prevent reporting old values
        self._data.object_com_acc_w[env_ids[:, None], object_ids] = 0.0

        # set into simulation
        view_ids = self._env_obj_ids_to_view_ids(env_ids, object_ids)
        self.root_physx_view.set_velocities(self.reshape_data_to_view(self._data.object_com_vel_w), indices=view_ids)

    def write_object_link_velocity_to_sim(
        self,
        object_velocity: torch.Tensor,
        env_ids: torch.Tensor | None = None,
        object_ids: slice | torch.Tensor | None = None,
    ):
        """Set the object link velocity over selected environment indices into the simulation.

        The velocity comprises linear velocity (x, y, z) and angular velocity (x, y, z) in that order.
        NOTE: This sets the velocity of the object's frame rather than the objects center of mass.

        Args:
            object_velocity: Object velocities in simulation frame. Shape is (len(env_ids), len(object_ids), 6).
            env_ids: Environment indices. If None, then all indices are used.
            object_ids: Object indices. If None, then all indices are used.
        """
        # resolve all indices
        # -- env_ids
        if env_ids is None:
            env_ids = self._ALL_ENV_INDICES
        # -- object_ids
        if object_ids is None:
            object_ids = self._ALL_OBJ_INDICES

        # set into internal buffers
        self._data.object_link_vel_w[env_ids[:, None], object_ids] = object_velocity.clone()
        # update these buffers only if the user is using them. Otherwise this adds to overhead.
        if self._data._object_link_state_w.data is not None:
            self._data.object_link_state_w[env_ids[:, None], object_ids, 7:] = object_velocity.clone()

        # get CoM pose in link frame
        quat = self.data.object_link_quat_w[env_ids[:, None], object_ids]
        com_pos_b = self.data.object_com_pos_b[env_ids[:, None], object_ids]
        # transform input velocity to center of mass frame
        object_com_velocity = object_velocity.clone()
        object_com_velocity[..., :3] += torch.linalg.cross(
            object_com_velocity[..., 3:], math_utils.quat_apply(quat, com_pos_b), dim=-1
        )

        # write center of mass velocity to sim
        self.write_object_com_velocity_to_sim(object_com_velocity, env_ids=env_ids, object_ids=object_ids)

    """
    Operations - Setters.
    """

    def set_external_force_and_torque(
        self,
        forces: torch.Tensor,
        torques: torch.Tensor,
        object_ids: slice | torch.Tensor | None = None,
        env_ids: torch.Tensor | None = None,
    ):
        """Set external force and torque to apply on the objects' bodies in their local frame.

        For many applications, we want to keep the applied external force on rigid bodies constant over a period of
        time (for instance, during the policy control). This function allows us to store the external force and torque
        into buffers which are then applied to the simulation at every step.

        .. caution::
            If the function is called with empty forces and torques, then this function disables the application
            of external wrench to the simulation.

            .. code-block:: python

                # example of disabling external wrench
                asset.set_external_force_and_torque(forces=torch.zeros(0, 0, 3), torques=torch.zeros(0, 0, 3))

        .. note::
            This function does not apply the external wrench to the simulation. It only fills the buffers with
            the desired values. To apply the external wrench, call the :meth:`write_data_to_sim` function
            right before the simulation step.

        Args:
            forces: External forces in bodies' local frame. Shape is (len(env_ids), len(object_ids), 3).
            torques: External torques in bodies' local frame. Shape is (len(env_ids), len(object_ids), 3).
            object_ids: Object indices to apply external wrench to. Defaults to None (all objects).
            env_ids: Environment indices to apply external wrench to. Defaults to None (all instances).
        """
        if forces.any() or torques.any():
            self.has_external_wrench = True
        else:
            self.has_external_wrench = False
            # to be safe, explicitly set value to zero
            forces = torques = 0.0

        # resolve all indices
        # -- env_ids
        if env_ids is None:
            env_ids = self._ALL_ENV_INDICES
        # -- object_ids
        if object_ids is None:
            object_ids = self._ALL_OBJ_INDICES
        # set into internal buffers
        self._external_force_b[env_ids[:, None], object_ids] = forces
        self._external_torque_b[env_ids[:, None], object_ids] = torques

    """
    Helper functions.
    """

    def reshape_view_to_data(self, data: torch.Tensor) -> torch.Tensor:
        """Reshapes and arranges the data coming from the :attr:`root_physx_view` to
        (num_instances, num_objects, data_dim).

        Args:
            data: The data coming from the :attr:`root_physx_view`. Shape is (num_instances * num_objects, data_dim).

        Returns:
            The reshaped data. Shape is (num_instances, num_objects, data_dim).
        """
        return torch.einsum("ijk -> jik", data.reshape(self.num_objects, self.num_instances, -1))

    def reshape_data_to_view(self, data: torch.Tensor) -> torch.Tensor:
        """Reshapes and arranges the data to the be consistent with data from the :attr:`root_physx_view`.

        Args:
            data: The data to be reshaped. Shape is (num_instances, num_objects, data_dim).

        Returns:
            The reshaped data. Shape is (num_instances * num_objects, data_dim).
        """
        return torch.einsum("ijk -> jik", data).reshape(self.num_objects * self.num_instances, *data.shape[2:])

    """
    Internal helper.
    """

    def _initialize_impl(self):
        # obtain global simulation view
        self._physics_sim_view = SimulationManager.get_physics_sim_view()
        root_prim_path_exprs = []
        for name, rigid_object_cfg in self.cfg.rigid_objects.items():
            # obtain the first prim in the regex expression (all others are assumed to be a copy of this)
            template_prim = sim_utils.find_first_matching_prim(rigid_object_cfg.prim_path)
            if template_prim is None:
                raise RuntimeError(f"Failed to find prim for expression: '{rigid_object_cfg.prim_path}'.")
            template_prim_path = template_prim.GetPath().pathString

            # find rigid root prims
            root_prims = sim_utils.get_all_matching_child_prims(
                template_prim_path, predicate=lambda prim: prim.HasAPI(UsdPhysics.RigidBodyAPI)
            )
            if len(root_prims) == 0:
                raise RuntimeError(
                    f"Failed to find a rigid body when resolving '{rigid_object_cfg.prim_path}'."
                    " Please ensure that the prim has 'USD RigidBodyAPI' applied."
                )
            if len(root_prims) > 1:
                raise RuntimeError(
                    f"Failed to find a single rigid body when resolving '{rigid_object_cfg.prim_path}'."
                    f" Found multiple '{root_prims}' under '{template_prim_path}'."
                    " Please ensure that there is only one rigid body in the prim path tree."
                )

            # check that no rigid object has an articulation root API, which decreases simulation performance
            articulation_prims = sim_utils.get_all_matching_child_prims(
                template_prim_path, predicate=lambda prim: prim.HasAPI(UsdPhysics.ArticulationRootAPI)
            )
            if len(articulation_prims) != 0:
                if articulation_prims[0].GetAttribute("physxArticulation:articulationEnabled").Get():
                    raise RuntimeError(
                        f"Found an articulation root when resolving '{rigid_object_cfg.prim_path}' in the rigid object"
                        f" collection. These are located at: '{articulation_prims}' under '{template_prim_path}'."
                        " Please disable the articulation root in the USD or from code by setting the parameter"
                        " 'ArticulationRootPropertiesCfg.articulation_enabled' to False in the spawn configuration."
                    )

            # resolve root prim back into regex expression
            root_prim_path = root_prims[0].GetPath().pathString
            root_prim_path_expr = rigid_object_cfg.prim_path + root_prim_path[len(template_prim_path) :]
            root_prim_path_exprs.append(root_prim_path_expr.replace(".*", "*"))

            self._object_names_list.append(name)

        # -- object view
        self._root_physx_view = self._physics_sim_view.create_rigid_body_view(root_prim_path_exprs)

        # check if the rigid body was created
        if self._root_physx_view._backend is None:
            raise RuntimeError("Failed to create rigid body collection. Please check PhysX logs.")

        # log information about the rigid body
        omni.log.info(f"Number of instances: {self.num_instances}")
        omni.log.info(f"Number of distinct objects: {self.num_objects}")
        omni.log.info(f"Object names: {self.object_names}")

        # container for data access
        self._data = RigidObjectCollectionData(self.root_physx_view, self.num_objects, self.device)

        # create buffers
        self._create_buffers()
        # process configuration
        self._process_cfg()
        # update the rigid body data
        self.update(0.0)

    def _create_buffers(self):
        """Create buffers for storing data."""
        # constants
        self._ALL_ENV_INDICES = torch.arange(self.num_instances, dtype=torch.long, device=self.device)
        self._ALL_OBJ_INDICES = torch.arange(self.num_objects, dtype=torch.long, device=self.device)

        # external forces and torques
        self.has_external_wrench = False
        self._external_force_b = torch.zeros((self.num_instances, self.num_objects, 3), device=self.device)
        self._external_torque_b = torch.zeros_like(self._external_force_b)

        # set information about rigid body into data
        self._data.object_names = self.object_names
        self._data.default_mass = self.reshape_view_to_data(self.root_physx_view.get_masses().clone())
        self._data.default_inertia = self.reshape_view_to_data(self.root_physx_view.get_inertias().clone())

    def _process_cfg(self):
        """Post processing of configuration parameters."""
        # default state
        # -- object state
        default_object_states = []
        for rigid_object_cfg in self.cfg.rigid_objects.values():
            default_object_state = (
                tuple(rigid_object_cfg.init_state.pos)
                + tuple(rigid_object_cfg.init_state.rot)
                + tuple(rigid_object_cfg.init_state.lin_vel)
                + tuple(rigid_object_cfg.init_state.ang_vel)
            )
            default_object_state = (
                torch.tensor(default_object_state, dtype=torch.float, device=self.device)
                .repeat(self.num_instances, 1)
                .unsqueeze(1)
            )
            default_object_states.append(default_object_state)
        # concatenate the default state for each object
        default_object_states = torch.cat(default_object_states, dim=1)
        self._data.default_object_state = default_object_states

    def _env_obj_ids_to_view_ids(
        self, env_ids: torch.Tensor, object_ids: Sequence[int] | slice | torch.Tensor
    ) -> torch.Tensor:
        """Converts environment and object indices to indices consistent with data from :attr:`root_physx_view`.

        Args:
            env_ids: Environment indices.
            object_ids: Object indices.

        Returns:
            The view indices.
        """
        # the order is env_0/object_0, env_0/object_1, env_0/object_..., env_1/object_0, env_1/object_1, ...
        # return a flat tensor of indices
        if isinstance(object_ids, slice):
            object_ids = self._ALL_OBJ_INDICES
        elif isinstance(object_ids, Sequence):
            object_ids = torch.tensor(object_ids, device=self.device)
        return (object_ids.unsqueeze(1) * self.num_instances + env_ids).flatten()

    """
    Internal simulation callbacks.
    """

    def _invalidate_initialize_callback(self, event):
        """Invalidates the scene elements."""
        # call parent
        super()._invalidate_initialize_callback(event)
        # set all existing views to None to invalidate them
        self._root_physx_view = None

    def _on_prim_deletion(self, prim_path: str) -> None:
        """Invalidates and deletes the callbacks when the prim is deleted.

        Args:
            prim_path: The path to the prim that is being deleted.

        Note:
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
