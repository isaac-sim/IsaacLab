# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Sequence

import carb
import omni.physics.tensors.impl.api as physx
from pxr import UsdPhysics

import omni.isaac.orbit.sim as sim_utils
import omni.isaac.orbit.utils.math as math_utils
import omni.isaac.orbit.utils.string as string_utils

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
    rigid body handle. This handle can be accessed using the :attr:`root_physx_view` and :attr:`body_physx_view`
    attributes. For a single rigid body asset, the :attr:`root_physx_view` and :attr:`body_physx_view` attributes
    are the same. However, these are different for articulated assets as explained in the :class:`Articulation`
    class.

    .. note::

        For users familiar with Isaac Sim, the PhysX view class API is not the exactly same as Isaac Sim view
        class API. Similar to Orbit, Isaac Sim wraps around the PhysX view API. However, as of now (2023.1 release),
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
        # container for data access
        self._data = RigidObjectData()

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
        """Number of bodies in the asset."""
        return 1

    @property
    def body_names(self) -> list[str]:
        """Ordered names of bodies in articulation."""
        prim_paths = self.body_physx_view.prim_paths[: self.num_bodies]
        return [path.split("/")[-1] for path in prim_paths]

    @property
    def root_physx_view(self) -> physx.RigidBodyView:
        """Rigid body view for the asset (PhysX).

        Note:
            Use this view with caution. It requires handling of tensors in a specific way.
        """
        return self._root_physx_view

    @property
    def body_physx_view(self) -> physx.RigidBodyView:
        """View for the bodies in the asset (PhysX).

        Note:
            Use this view with caution. It requires handling of tensors in a specific way.
        """
        return self._body_physx_view

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
        # reset last body vel
        self._last_body_vel_w[env_ids] = 0.0

    def write_data_to_sim(self):
        """Write external wrench to the simulation.

        Note:
            We write external wrench to the simulation here since this function is called before the simulation step.
            This ensures that the external wrench is applied at every simulation step.
        """
        # write external wrench
        if self.has_external_wrench:
            self.body_physx_view.apply_forces_and_torques_at_position(
                force_data=self._external_force_b.view(-1, 3),
                torque_data=self._external_torque_b.view(-1, 3),
                position_data=None,
                indices=self._ALL_BODY_INDICES,
                is_global=False,
            )

    def update(self, dt: float):
        # -- root-state (note: we roll the quaternion to match the convention used in Isaac Sim -- wxyz)
        self._data.root_state_w[:, :7] = self.root_physx_view.get_transforms()
        self._data.root_state_w[:, 3:7] = math_utils.convert_quat(self._data.root_state_w[:, 3:7], to="wxyz")
        self._data.root_state_w[:, 7:] = self.root_physx_view.get_velocities()
        # -- update common data
        self._update_common_data(dt)

    def find_bodies(self, name_keys: str | Sequence[str]) -> tuple[list[int], list[str]]:
        """Find bodies in the articulation based on the name keys.

        Please check the :meth:`omni.isaac.orbit.utils.string_utils.resolve_matching_names` function for more
        information on the name matching.

        Args:
            name_keys: A regular expression or a list of regular expressions to match the body names.

        Returns:
            A tuple of lists containing the body indices and names.
        """
        return string_utils.resolve_matching_names(name_keys, self.body_names)

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
        # set into simulation
        self.write_root_pose_to_sim(root_state[:, :7], env_ids=env_ids)
        self.write_root_velocity_to_sim(root_state[:, 7:], env_ids=env_ids)

    def write_root_pose_to_sim(self, root_pose: torch.Tensor, env_ids: Sequence[int] | None = None):
        """Set the root pose over selected environment indices into the simulation.

        The root pose comprises of the cartesian position and quaternion orientation in (w, x, y, z).

        Args:
            root_pose: Root poses in simulation frame. Shape is (len(env_ids), 7).
            env_ids: Environment indices. If None, then all indices are used.
        """
        # resolve all indices
        physx_env_ids = env_ids
        if env_ids is None:
            env_ids = slice(None)
            physx_env_ids = self._ALL_INDICES
        # note: we need to do this here since tensors are not set into simulation until step.
        # set into internal buffers
        self._data.root_state_w[env_ids, :7] = root_pose.clone()
        # convert root quaternion from wxyz to xyzw
        root_poses_xyzw = self._data.root_state_w[:, :7].clone()
        root_poses_xyzw[:, 3:] = math_utils.convert_quat(root_poses_xyzw[:, 3:], to="xyzw")
        # set into simulation
        self.root_physx_view.set_transforms(root_poses_xyzw, indices=physx_env_ids)

    def write_root_velocity_to_sim(self, root_velocity: torch.Tensor, env_ids: Sequence[int] | None = None):
        """Set the root velocity over selected environment indices into the simulation.

        Args:
            root_velocity: Root velocities in simulation frame. Shape is (len(env_ids), 6).
            env_ids: Environment indices. If None, then all indices are used.
        """
        # resolve all indices
        physx_env_ids = env_ids
        if env_ids is None:
            env_ids = slice(None)
            physx_env_ids = self._ALL_INDICES
        # note: we need to do this here since tensors are not set into simulation until step.
        # set into internal buffers
        self._data.root_state_w[env_ids, 7:] = root_velocity.clone()
        # set into simulation
        self.root_physx_view.set_velocities(self._data.root_state_w[:, 7:], indices=physx_env_ids)

    """
    Operations - Setters.
    """

    def set_external_force_and_torque(
        self,
        forces: torch.Tensor,
        torques: torch.Tensor,
        body_ids: Sequence[int] | None = None,
        env_ids: Sequence[int] | None = None,
    ):
        """Set external force and torque to apply on the asset's bodies in their local frame.

        For many applications, we want to keep the applied external force on rigid bodies constant over a period of
        time (for instance, during the policy control). This function allows us to store the external force and torque
        into buffers which are then applied to the simulation at every step.

        .. caution::
            If the function is called with empty forces and torques, then this function disables the application
            of external wrench to the simulation.

            .. code-block:: python

                # example of disabling external wrench
                asset.set_external_force_and_torque(forces=torch.zeros(0, 3), torques=torch.zeros(0, 3))

        .. note::
            This function does not apply the external wrench to the simulation. It only fills the buffers with
            the desired values. To apply the external wrench, call the :meth:`write_data_to_sim` function.

        Args:
            forces: External forces in bodies' local frame. Shape is (len(env_ids), len(body_ids), 3).
            torques: External torques in bodies' local frame. Shape is (len(env_ids), len(body_ids), 3).
            body_ids: Body indices to apply external wrench to. Defaults to None (all bodies).
            env_ids: Environment indices to apply external wrench to. Defaults to None (all instances).
        """
        if forces.any() or torques.any():
            self.has_external_wrench = True
            # resolve all indices
            # -- env_ids
            if env_ids is None:
                env_ids = self._ALL_INDICES
            elif not isinstance(env_ids, torch.Tensor):
                env_ids = torch.tensor(env_ids, dtype=torch.long, device=self.device)
            # -- body_ids
            if body_ids is None:
                body_ids = torch.arange(self.num_bodies, dtype=torch.long, device=self.device)
            elif not isinstance(body_ids, torch.Tensor):
                body_ids = torch.tensor(body_ids, dtype=torch.long, device=self.device)

            # note: we need to do this complicated indexing since torch doesn't support multi-indexing
            # create global body indices from env_ids and env_body_ids
            # (env_id * total_bodies_per_env) + body_id
            total_bodies_per_env = self.body_physx_view.count // self.root_physx_view.count
            indices = body_ids.repeat(len(env_ids), 1) + env_ids.unsqueeze(1) * total_bodies_per_env
            indices = indices.view(-1)
            # set into internal buffers
            # note: these are applied in the write_to_sim function
            self._external_force_b.flatten(0, 1)[indices] = forces.flatten(0, 1)
            self._external_torque_b.flatten(0, 1)[indices] = torques.flatten(0, 1)
        else:
            self.has_external_wrench = False

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
        # find rigid root prims
        root_prims = sim_utils.get_all_matching_child_prims(
            template_prim_path, predicate=lambda prim: prim.HasAPI(UsdPhysics.RigidBodyAPI)
        )
        if len(root_prims) != 1:
            raise RuntimeError(
                f"Failed to find a single rigid body when resolving '{self.cfg.prim_path}'."
                f" Found multiple '{root_prims}' under '{template_prim_path}'."
            )
        # resolve root prim back into regex expression
        root_prim_path = root_prims[0].GetPath().pathString
        root_prim_path_expr = self.cfg.prim_path + root_prim_path[len(template_prim_path) :]
        # -- object view
        self._root_physx_view = self._physics_sim_view.create_rigid_body_view(root_prim_path_expr.replace(".*", "*"))
        self._body_physx_view = self._root_physx_view
        # log information about the articulation
        carb.log_info(f"Rigid body initialized at: {self.cfg.prim_path} with root '{root_prim_path_expr}'.")
        carb.log_info(f"Number of instances: {self.num_instances}")
        carb.log_info(f"Number of bodies: {self.num_bodies}")
        carb.log_info(f"Body names: {self.body_names}")
        # create buffers
        self._create_buffers()
        # process configuration
        self._process_cfg()

    def _create_buffers(self):
        """Create buffers for storing data."""
        # constants
        self._ALL_INDICES = torch.arange(self.num_instances, dtype=torch.long, device=self.device)
        self._ALL_BODY_INDICES = torch.arange(self.body_physx_view.count, dtype=torch.long, device=self.device)
        self.GRAVITY_VEC_W = torch.tensor((0.0, 0.0, -1.0), device=self.device).repeat(self.num_instances, 1)
        self.FORWARD_VEC_B = torch.tensor((1.0, 0.0, 0.0), device=self.device).repeat(self.num_instances, 1)
        # external forces and torques
        self.has_external_wrench = False
        self._external_force_b = torch.zeros((self.num_instances, self.num_bodies, 3), device=self.device)
        self._external_torque_b = torch.zeros_like(self._external_force_b)

        # asset data
        # -- properties
        self._data.body_names = self.body_names
        # -- root states
        self._data.root_state_w = torch.zeros(self.num_instances, 13, device=self.device)
        self._data.default_root_state = torch.zeros_like(self._data.root_state_w)
        # -- body states
        self._data.body_state_w = torch.zeros(self.num_instances, self.num_bodies, 13, device=self.device)
        # -- post-computed
        self._data.root_vel_b = torch.zeros(self.num_instances, 6, device=self.device)
        self._data.projected_gravity_b = torch.zeros(self.num_instances, 3, device=self.device)
        self._data.heading_w = torch.zeros(self.num_instances, device=self.device)
        self._data.body_acc_w = torch.zeros(self.num_instances, self.num_bodies, 6, device=self.device)

        # history buffers for quantities
        # -- used to compute body accelerations numerically
        self._last_body_vel_w = torch.zeros(self.num_instances, self.num_bodies, 6, device=self.device)

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

    def _update_common_data(self, dt: float):
        """Update common quantities related to rigid objects.

        Note:
            This has been separated from the update function to allow for the child classes to
            override the update function without having to worry about updating the common data.
        """
        # -- body-state (note: we roll the quaternion to match the convention used in Isaac Sim -- wxyz)
        self._data.body_state_w[..., :7] = self.body_physx_view.get_transforms().view(-1, self.num_bodies, 7)
        self._data.body_state_w[..., 3:7] = math_utils.convert_quat(self._data.body_state_w[..., 3:7], to="wxyz")
        self._data.body_state_w[..., 7:] = self.body_physx_view.get_velocities().view(-1, self.num_bodies, 6)
        # -- body acceleration
        self._data.body_acc_w[:] = (self._data.body_state_w[..., 7:] - self._last_body_vel_w) / dt
        self._last_body_vel_w[:] = self._data.body_state_w[..., 7:]
        # -- root state in body frame
        self._data.root_vel_b[:, 0:3] = math_utils.quat_rotate_inverse(
            self._data.root_quat_w, self._data.root_lin_vel_w
        )
        self._data.root_vel_b[:, 3:6] = math_utils.quat_rotate_inverse(
            self._data.root_quat_w, self._data.root_ang_vel_w
        )
        self._data.projected_gravity_b[:] = math_utils.quat_rotate_inverse(self._data.root_quat_w, self.GRAVITY_VEC_W)
        # -- heading direction of root
        forward_w = math_utils.quat_apply(self._data.root_quat_w, self.FORWARD_VEC_B)
        self._data.heading_w[:] = torch.atan2(forward_w[:, 1], forward_w[:, 0])

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
        self._body_physx_view = None
