# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import omni.log
import omni.physics.tensors.impl.api as physx
from isaacsim.core.simulation_manager import SimulationManager
from pxr import UsdPhysics

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
import isaaclab.utils.string as string_utils

from ..asset_base import AssetBase
from .rigid_object_data import RigidObjectData

if TYPE_CHECKING:
    from .rigid_object_cfg import RigidObjectCfg


class RigidObject(AssetBase):
    """A rigid object asset class.

    Rigid objects are assets comprising of rigid bodies. They can be used to represent dynamic objects
    such as boxes, spheres, etc. A rigid body is described by its pose, velocity and mass distribution.

    For an asset to be considered a rigid object, the root prim of the asset must have the `USD RigidBodyAPI`_
    applied to it. This API is used to define the simulation properties of the rigid body. On playing the
    simulation, the physics engine will automatically register the rigid body and create a corresponding
    rigid body handle. This handle can be accessed using the :attr:`root_physx_view` attribute.

    .. note::

        For users familiar with Isaac Sim, the PhysX view class API is not the exactly same as Isaac Sim view
        class API. Similar to Isaac Lab, Isaac Sim wraps around the PhysX view API. However, as of now (2023.1 release),
        we see a large difference in initializing the view classes in Isaac Sim. This is because the view classes
        in Isaac Sim perform additional USD-related operations which are slow and also not required.

    .. _`USD RigidBodyAPI`: https://openusd.org/dev/api/class_usd_physics_rigid_body_a_p_i.html
    """

    cfg: RigidObjectCfg
    """Configuration instance for the rigid object."""

    def __init__(self, cfg: RigidObjectCfg):
        """Initialize the rigid object.

        Args:
            cfg: A configuration instance.
        """
        super().__init__(cfg)

    """
    Properties
    """

    @property
    def data(self) -> RigidObjectData:
        return self._data

    @property
    def num_instances(self) -> int:
        return self.root_physx_view.count

    @property
    def num_bodies(self) -> int:
        """Number of bodies in the asset.

        This is always 1 since each object is a single rigid body.
        """
        return 1

    @property
    def body_names(self) -> list[str]:
        """Ordered names of bodies in the rigid object."""
        prim_paths = self.root_physx_view.prim_paths[: self.num_bodies]
        return [path.split("/")[-1] for path in prim_paths]

    @property
    def root_physx_view(self) -> physx.RigidBodyView:
        """Rigid body view for the asset (PhysX).

        Note:
            Use this view with caution. It requires handling of tensors in a specific way.
        """
        return self._root_physx_view

    """
    Operations.
    """

    def reset(self, env_ids: Sequence[int] | None = None):
        # resolve all indices
        if env_ids is None:
            env_ids = slice(None)
        # reset external wrench
        self._external_force_b[env_ids] = 0.0
        self._external_torque_b[env_ids] = 0.0
        self._external_wrench_positions_b[env_ids] = 0.0

    def write_data_to_sim(self):
        """Write external wrench to the simulation.

        Note:
            We write external wrench to the simulation here since this function is called before the simulation step.
            This ensures that the external wrench is applied at every simulation step.
        """
        # write external wrench
        if self.has_external_wrench:
            if self.uses_external_wrench_positions:
                self.root_physx_view.apply_forces_and_torques_at_position(
                    force_data=self._external_force_b.view(-1, 3),
                    torque_data=self._external_torque_b.view(-1, 3),
                    position_data=self._external_wrench_positions_b.view(-1, 3),
                    indices=self._ALL_INDICES,
                    is_global=self._use_global_wrench_frame,
                )
            else:
                self.root_physx_view.apply_forces_and_torques_at_position(
                    force_data=self._external_force_b.view(-1, 3),
                    torque_data=self._external_torque_b.view(-1, 3),
                    position_data=None,
                    indices=self._ALL_INDICES,
                    is_global=self._use_global_wrench_frame,
                )

    def update(self, dt: float):
        self._data.update(dt)

    """
    Operations - Finders.
    """

    def find_bodies(self, name_keys: str | Sequence[str], preserve_order: bool = False) -> tuple[list[int], list[str]]:
        """Find bodies in the rigid body based on the name keys.

        Please check the :meth:`isaaclab.utils.string_utils.resolve_matching_names` function for more
        information on the name matching.

        Args:
            name_keys: A regular expression or a list of regular expressions to match the body names.
            preserve_order: Whether to preserve the order of the name keys in the output. Defaults to False.

        Returns:
            A tuple of lists containing the body indices and names.
        """
        return string_utils.resolve_matching_names(name_keys, self.body_names, preserve_order)

    """
    Operations - Write to simulation.
    """

    def write_root_state_to_sim(self, root_state: torch.Tensor, env_ids: Sequence[int] | None = None):
        """Set the root state over selected environment indices into the simulation.

        The root state comprises of the cartesian position, quaternion orientation in (w, x, y, z), and linear
        and angular velocity. All the quantities are in the simulation frame.

        Args:
            root_state: Root state in simulation frame. Shape is (len(env_ids), 13).
            env_ids: Environment indices. If None, then all indices are used.
        """
        self.write_root_link_pose_to_sim(root_state[:, :7], env_ids=env_ids)
        self.write_root_com_velocity_to_sim(root_state[:, 7:], env_ids=env_ids)

    def write_root_com_state_to_sim(self, root_state: torch.Tensor, env_ids: Sequence[int] | None = None):
        """Set the root center of mass state over selected environment indices into the simulation.

        The root state comprises of the cartesian position, quaternion orientation in (w, x, y, z), and linear
        and angular velocity. All the quantities are in the simulation frame.

        Args:
            root_state: Root state in simulation frame. Shape is (len(env_ids), 13).
            env_ids: Environment indices. If None, then all indices are used.
        """
        self.write_root_com_pose_to_sim(root_state[:, :7], env_ids=env_ids)
        self.write_root_com_velocity_to_sim(root_state[:, 7:], env_ids=env_ids)

    def write_root_link_state_to_sim(self, root_state: torch.Tensor, env_ids: Sequence[int] | None = None):
        """Set the root link state over selected environment indices into the simulation.

        The root state comprises of the cartesian position, quaternion orientation in (w, x, y, z), and linear
        and angular velocity. All the quantities are in the simulation frame.

        Args:
            root_state: Root state in simulation frame. Shape is (len(env_ids), 13).
            env_ids: Environment indices. If None, then all indices are used.
        """
        self.write_root_link_pose_to_sim(root_state[:, :7], env_ids=env_ids)
        self.write_root_link_velocity_to_sim(root_state[:, 7:], env_ids=env_ids)

    def write_root_pose_to_sim(self, root_pose: torch.Tensor, env_ids: Sequence[int] | None = None):
        """Set the root pose over selected environment indices into the simulation.

        The root pose comprises of the cartesian position and quaternion orientation in (w, x, y, z).

        Args:
            root_pose: Root link poses in simulation frame. Shape is (len(env_ids), 7).
            env_ids: Environment indices. If None, then all indices are used.
        """
        self.write_root_link_pose_to_sim(root_pose, env_ids=env_ids)

    def write_root_link_pose_to_sim(self, root_pose: torch.Tensor, env_ids: Sequence[int] | None = None):
        """Set the root link pose over selected environment indices into the simulation.

        The root pose comprises of the cartesian position and quaternion orientation in (w, x, y, z).

        Args:
            root_pose: Root link poses in simulation frame. Shape is (len(env_ids), 7).
            env_ids: Environment indices. If None, then all indices are used.
        """
        # resolve all indices
        physx_env_ids = env_ids
        if env_ids is None:
            env_ids = slice(None)
            physx_env_ids = self._ALL_INDICES

        # note: we need to do this here since tensors are not set into simulation until step.
        # set into internal buffers
        self._data.root_link_pose_w[env_ids] = root_pose.clone()
        # update these buffers only if the user is using them. Otherwise this adds to overhead.
        if self._data._root_link_state_w.data is not None:
            self._data.root_link_state_w[env_ids, :7] = self._data.root_link_pose_w[env_ids]
        if self._data._root_state_w.data is not None:
            self._data.root_state_w[env_ids, :7] = self._data.root_link_pose_w[env_ids]
        if self._data._root_com_state_w.data is not None:
            expected_com_pos, expected_com_quat = math_utils.combine_frame_transforms(
                self._data.root_link_pose_w[env_ids, :3],
                self._data.root_link_pose_w[env_ids, 3:7],
                self.data.body_com_pos_b[env_ids, 0, :],
                self.data.body_com_quat_b[env_ids, 0, :],
            )
            self._data.root_com_state_w[env_ids, :3] = expected_com_pos
            self._data.root_com_state_w[env_ids, 3:7] = expected_com_quat
        # convert root quaternion from wxyz to xyzw
        root_poses_xyzw = self._data.root_link_pose_w.clone()
        root_poses_xyzw[:, 3:] = math_utils.convert_quat(root_poses_xyzw[:, 3:], to="xyzw")
        # set into simulation
        self.root_physx_view.set_transforms(root_poses_xyzw, indices=physx_env_ids)

    def write_root_com_pose_to_sim(self, root_pose: torch.Tensor, env_ids: Sequence[int] | None = None):
        """Set the root center of mass pose over selected environment indices into the simulation.

        The root pose comprises of the cartesian position and quaternion orientation in (w, x, y, z).
        The orientation is the orientation of the principle axes of inertia.

        Args:
            root_pose: Root center of mass poses in simulation frame. Shape is (len(env_ids), 7).
            env_ids: Environment indices. If None, then all indices are used.
        """
        # resolve all indices
        if env_ids is None:
            local_env_ids = slice(env_ids)
        else:
            local_env_ids = env_ids

        # set into internal buffers
        self._data.root_com_pose_w[local_env_ids] = root_pose.clone()
        # update these buffers only if the user is using them. Otherwise this adds to overhead.
        if self._data._root_com_state_w.data is not None:
            self._data.root_com_state_w[local_env_ids, :7] = self._data.root_com_pose_w[local_env_ids]

        # get CoM pose in link frame
        com_pos_b = self.data.body_com_pos_b[local_env_ids, 0, :]
        com_quat_b = self.data.body_com_quat_b[local_env_ids, 0, :]
        # transform input CoM pose to link frame
        root_link_pos, root_link_quat = math_utils.combine_frame_transforms(
            root_pose[..., :3],
            root_pose[..., 3:7],
            math_utils.quat_apply(math_utils.quat_inv(com_quat_b), -com_pos_b),
            math_utils.quat_inv(com_quat_b),
        )
        root_link_pose = torch.cat((root_link_pos, root_link_quat), dim=-1)

        # write transformed pose in link frame to sim
        self.write_root_link_pose_to_sim(root_link_pose, env_ids=env_ids)

    def write_root_velocity_to_sim(self, root_velocity: torch.Tensor, env_ids: Sequence[int] | None = None):
        """Set the root center of mass velocity over selected environment indices into the simulation.

        The velocity comprises linear velocity (x, y, z) and angular velocity (x, y, z) in that order.
        NOTE: This sets the velocity of the root's center of mass rather than the roots frame.

        Args:
            root_velocity: Root center of mass velocities in simulation world frame. Shape is (len(env_ids), 6).
            env_ids: Environment indices. If None, then all indices are used.
        """
        self.write_root_com_velocity_to_sim(root_velocity=root_velocity, env_ids=env_ids)

    def write_root_com_velocity_to_sim(self, root_velocity: torch.Tensor, env_ids: Sequence[int] | None = None):
        """Set the root center of mass velocity over selected environment indices into the simulation.

        The velocity comprises linear velocity (x, y, z) and angular velocity (x, y, z) in that order.
        NOTE: This sets the velocity of the root's center of mass rather than the roots frame.

        Args:
            root_velocity: Root center of mass velocities in simulation world frame. Shape is (len(env_ids), 6).
            env_ids: Environment indices. If None, then all indices are used.
        """
        # resolve all indices
        physx_env_ids = env_ids
        if env_ids is None:
            env_ids = slice(None)
            physx_env_ids = self._ALL_INDICES

        # note: we need to do this here since tensors are not set into simulation until step.
        # set into internal buffers
        self._data.root_com_vel_w[env_ids] = root_velocity.clone()
        # update these buffers only if the user is using them. Otherwise this adds to overhead.
        if self._data._root_com_state_w.data is not None:
            self._data.root_com_state_w[env_ids, 7:] = self._data.root_com_vel_w[env_ids]
        if self._data._root_state_w.data is not None:
            self._data.root_state_w[env_ids, 7:] = self._data.root_com_vel_w[env_ids]
        if self._data._root_link_state_w.data is not None:
            self._data.root_link_state_w[env_ids, 7:] = self._data.root_com_vel_w[env_ids]
        # make the acceleration zero to prevent reporting old values
        self._data.body_com_acc_w[env_ids] = 0.0
        # set into simulation
        self.root_physx_view.set_velocities(self._data.root_com_vel_w, indices=physx_env_ids)

    def write_root_link_velocity_to_sim(self, root_velocity: torch.Tensor, env_ids: Sequence[int] | None = None):
        """Set the root link velocity over selected environment indices into the simulation.

        The velocity comprises linear velocity (x, y, z) and angular velocity (x, y, z) in that order.
        NOTE: This sets the velocity of the root's frame rather than the roots center of mass.

        Args:
            root_velocity: Root frame velocities in simulation world frame. Shape is (len(env_ids), 6).
            env_ids: Environment indices. If None, then all indices are used.
        """
        # resolve all indices
        if env_ids is None:
            local_env_ids = slice(env_ids)
        else:
            local_env_ids = env_ids

        # set into internal buffers
        self._data.root_link_vel_w[local_env_ids] = root_velocity.clone()
        # update these buffers only if the user is using them. Otherwise this adds to overhead.
        if self._data._root_link_state_w.data is not None:
            self._data.root_link_state_w[local_env_ids, 7:] = self._data.root_link_vel_w[local_env_ids]

        # get CoM pose in link frame
        quat = self.data.root_link_quat_w[local_env_ids]
        com_pos_b = self.data.body_com_pos_b[local_env_ids, 0, :]
        # transform input velocity to center of mass frame
        root_com_velocity = root_velocity.clone()
        root_com_velocity[:, :3] += torch.linalg.cross(
            root_com_velocity[:, 3:], math_utils.quat_apply(quat, com_pos_b), dim=-1
        )

        # write transformed velocity in CoM frame to sim
        self.write_root_com_velocity_to_sim(root_com_velocity, env_ids=env_ids)

    """
    Operations - Setters.
    """

    def set_external_force_and_torque(
        self,
        forces: torch.Tensor,
        torques: torch.Tensor,
        positions: torch.Tensor | None = None,
        body_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | None = None,
        is_global: bool = False,
    ):
        """Set external force and torque to apply on the asset's bodies in their local frame.

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
            positions: External wrench positions in bodies' local frame. Shape is (len(env_ids), len(body_ids), 3). Defaults to None.
            body_ids: Body indices to apply external wrench to. Defaults to None (all bodies).
            env_ids: Environment indices to apply external wrench to. Defaults to None (all instances).
            is_global: Whether to apply the external wrench in the global frame. Defaults to False. If set to False,
                the external wrench is applied in the link frame of the bodies.
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
            env_ids = slice(None)
        # -- body_ids
        if body_ids is None:
            body_ids = slice(None)
        # broadcast env_ids if needed to allow double indexing
        if env_ids != slice(None) and body_ids != slice(None):
            env_ids = env_ids[:, None]
        # set into internal buffers
        self._external_force_b[env_ids, body_ids] = forces
        self._external_torque_b[env_ids, body_ids] = torques

        if is_global != self._use_global_wrench_frame:
            omni.log.warn(
                f"The external wrench frame has been changed from {self._use_global_wrench_frame} to {is_global}. This"
                " may lead to unexpected behavior."
            )
            self._use_global_wrench_frame = is_global

        if positions is not None:
            self.uses_external_wrench_positions = True
            self._external_wrench_positions_b[env_ids, body_ids] = positions
        else:
            if self.uses_external_wrench_positions:
                self._external_wrench_positions_b[env_ids, body_ids] = 0.0

    """
    Internal helper.
    """

    def _initialize_impl(self):
        # obtain global simulation view
        self._physics_sim_view = SimulationManager.get_physics_sim_view()
        # obtain the first prim in the regex expression (all others are assumed to be a copy of this)
        template_prim = sim_utils.find_first_matching_prim(self.cfg.prim_path)
        if template_prim is None:
            raise RuntimeError(f"Failed to find prim for expression: '{self.cfg.prim_path}'.")
        template_prim_path = template_prim.GetPath().pathString

        # find rigid root prims
        root_prims = sim_utils.get_all_matching_child_prims(
            template_prim_path, predicate=lambda prim: prim.HasAPI(UsdPhysics.RigidBodyAPI)
        )
        if len(root_prims) == 0:
            raise RuntimeError(
                f"Failed to find a rigid body when resolving '{self.cfg.prim_path}'."
                " Please ensure that the prim has 'USD RigidBodyAPI' applied."
            )
        if len(root_prims) > 1:
            raise RuntimeError(
                f"Failed to find a single rigid body when resolving '{self.cfg.prim_path}'."
                f" Found multiple '{root_prims}' under '{template_prim_path}'."
                " Please ensure that there is only one rigid body in the prim path tree."
            )

        articulation_prims = sim_utils.get_all_matching_child_prims(
            template_prim_path, predicate=lambda prim: prim.HasAPI(UsdPhysics.ArticulationRootAPI)
        )
        if len(articulation_prims) != 0:
            if articulation_prims[0].GetAttribute("physxArticulation:articulationEnabled").Get():
                raise RuntimeError(
                    f"Found an articulation root when resolving '{self.cfg.prim_path}' for rigid objects. These are"
                    f" located at: '{articulation_prims}' under '{template_prim_path}'. Please disable the articulation"
                    " root in the USD or from code by setting the parameter"
                    " 'ArticulationRootPropertiesCfg.articulation_enabled' to False in the spawn configuration."
                )

        # resolve root prim back into regex expression
        root_prim_path = root_prims[0].GetPath().pathString
        root_prim_path_expr = self.cfg.prim_path + root_prim_path[len(template_prim_path) :]
        # -- object view
        self._root_physx_view = self._physics_sim_view.create_rigid_body_view(root_prim_path_expr.replace(".*", "*"))

        # check if the rigid body was created
        if self._root_physx_view._backend is None:
            raise RuntimeError(f"Failed to create rigid body at: {self.cfg.prim_path}. Please check PhysX logs.")

        # log information about the rigid body
        omni.log.info(f"Rigid body initialized at: {self.cfg.prim_path} with root '{root_prim_path_expr}'.")
        omni.log.info(f"Number of instances: {self.num_instances}")
        omni.log.info(f"Number of bodies: {self.num_bodies}")
        omni.log.info(f"Body names: {self.body_names}")

        # container for data access
        self._data = RigidObjectData(self.root_physx_view, self.device)

        # create buffers
        self._create_buffers()
        # process configuration
        self._process_cfg()
        # update the rigid body data
        self.update(0.0)

    def _create_buffers(self):
        """Create buffers for storing data."""
        # constants
        self._ALL_INDICES = torch.arange(self.num_instances, dtype=torch.long, device=self.device)

        # external forces and torques
        self.has_external_wrench = False
        self._external_force_b = torch.zeros((self.num_instances, self.num_bodies, 3), device=self.device)
        self._external_torque_b = torch.zeros_like(self._external_force_b)
        self.uses_external_wrench_positions = False
        self._external_wrench_positions_b = torch.zeros_like(self._external_force_b)
        self._use_global_wrench_frame = False

        # set information about rigid body into data
        self._data.body_names = self.body_names
        self._data.default_mass = self.root_physx_view.get_masses().clone()
        self._data.default_inertia = self.root_physx_view.get_inertias().clone()

    def _process_cfg(self):
        """Post processing of configuration parameters."""
        # default state
        # -- root state
        # note: we cast to tuple to avoid torch/numpy type mismatch.
        default_root_state = (
            tuple(self.cfg.init_state.pos)
            + tuple(self.cfg.init_state.rot)
            + tuple(self.cfg.init_state.lin_vel)
            + tuple(self.cfg.init_state.ang_vel)
        )
        default_root_state = torch.tensor(default_root_state, dtype=torch.float, device=self.device)
        self._data.default_root_state = default_root_state.repeat(self.num_instances, 1)

    """
    Internal simulation callbacks.
    """

    def _invalidate_initialize_callback(self, event):
        """Invalidates the scene elements."""
        # call parent
        super()._invalidate_initialize_callback(event)
        # set all existing views to None to invalidate them
        self._root_physx_view = None
