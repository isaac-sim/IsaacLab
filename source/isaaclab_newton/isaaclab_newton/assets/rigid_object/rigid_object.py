# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import logging
import torch
import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING

import warp as wp
from isaaclab_newton.assets.rigid_object.rigid_object_data import RigidObjectData
from isaaclab_newton.assets.utils.shared import find_bodies
from isaaclab_newton.kernels import (
    project_link_velocity_to_com_frame_masked_root,
    split_state_to_pose_and_velocity,
    transform_CoM_pose_to_link_frame_masked_root,
    update_wrench_array_with_force,
    update_wrench_array_with_torque,
    vec13f,
)
from newton.selection import ArticulationView as NewtonArticulationView
from newton.solvers import SolverNotifyFlags
from pxr import UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.assets.rigid_object.base_rigid_object import BaseRigidObject
from isaaclab.sim._impl.newton_manager import NewtonManager
from isaaclab.utils.helpers import deprecated
from isaaclab.utils.warp.update_kernels import (
    update_array1D_with_array1D_masked,
    update_array1D_with_value,
    update_array2D_with_array2D_masked,
    update_array2D_with_value_masked,
)
from isaaclab.utils.warp.utils import (
    make_complete_data_from_torch_dual_index,
    make_complete_data_from_torch_single_index,
    make_masks_from_torch_ids,
)

if TYPE_CHECKING:
    from isaaclab.assets.rigid_object.rigid_object_cfg import RigidObjectCfg

logger = logging.getLogger(__name__)
warnings.simplefilter("once", UserWarning)
logging.captureWarnings(True)


class RigidObject(BaseRigidObject):
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

    __backend_name__: str = "newton"
    """The name of the backend for the rigid object."""

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
        return self._root_view.count

    @property
    def num_bodies(self) -> int:
        """Number of bodies in the asset.

        This is always 1 since each object is a single rigid body.
        """
        return self._root_view.link_count

    @property
    def num_shapes_per_body(self) -> list[int]:
        """Number of collision shapes per body in the articulation.

        This property returns a list where each element represents the number of collision
        shapes for the corresponding body in the articulation. This is cached for efficient
        access during material property randomization and other operations.

        Returns:
            List of integers representing the number of shapes per body.
        """
        if not hasattr(self, "_num_shapes_per_body"):
            self._num_shapes_per_body = []
            for shapes in self._root_view.body_shapes:
                self._num_shapes_per_body.append(len(shapes))
        return self._num_shapes_per_body

    @property
    def body_names(self) -> list[str]:
        """Ordered names of bodies in the rigid object."""
        return self._root_view.body_names

    @property
    def root_view(self):
        """Root view for the asset.

        Note:
            Use this view with caution. It requires handling of tensors in a specific way.
        """
        return self._root_view

    """
    Operations.
    """

    def reset(self, ids: Sequence[int] | None = None, mask: wp.array | None = None):
        if ids is not None and mask is None:
            mask = torch.zeros(self.num_instances, dtype=torch.bool, device=self.device)
            mask[ids] = True
            mask = wp.from_torch(mask, dtype=wp.bool)
        elif mask is not None:
            if isinstance(mask, torch.Tensor):
                mask = wp.from_torch(mask, dtype=wp.bool)
        else:
            mask = self._data.ALL_ENV_MASK
        # reset external wrench
        self.has_external_wrench = False
        wp.launch(
            update_array2D_with_value_masked,
            dim=(self.num_instances, self.num_bodies),
            device=self.device,
            inputs=[
                wp.vec3f(0.0, 0.0, 0.0),
                self._external_force_b,
                mask,
                self._data.ALL_BODY_MASK,
            ],
        )
        wp.launch(
            update_array2D_with_value_masked,
            dim=(self.num_instances, self.num_bodies),
            device=self.device,
            inputs=[
                wp.vec3f(0.0, 0.0, 0.0),
                self._external_torque_b,
                mask,
                self._data.ALL_BODY_MASK,
            ],
        )

    def write_data_to_sim(self) -> None:
        """Write external wrench to the simulation.

        Note:
            We write external wrench to the simulation here since this function is called before the simulation step.
            This ensures that the external wrench is applied at every simulation step.
        """
        # FIXME: This is a temporary solution to write the external wrench to the simulation.
        # We could use slicing instead, so this comes for free? Or a single update at least?
        if self.has_external_wrench:
            wp.launch(
                update_wrench_array_with_force,
                dim=(self.num_instances, self.num_bodies),
                device=self.device,
                inputs=[
                    self._external_force_b,
                    self._data._sim_bind_body_external_wrench,
                    self._data.ALL_ENV_MASK,
                    self._data.ALL_BODY_MASK,
                ],
            )
            wp.launch(
                update_wrench_array_with_torque,
                dim=(self.num_instances, self.num_bodies),
                device=self.device,
                inputs=[
                    self._external_torque_b,
                    self._data._sim_bind_body_external_wrench,
                    self._data.ALL_ENV_MASK,
                    self._data.ALL_BODY_MASK,
                ],
            )

    def update(self, dt: float) -> None:
        self._data.update(dt)

    """
    Operations - Finders.
    """

    def find_bodies(
        self, name_keys: str | Sequence[str], preserve_order: bool = False
    ) -> tuple[wp.array, list[str], list[int]]:
        """Find bodies in the rigid body based on the name keys.

        Please check the :meth:`isaaclab.utils.string_utils.resolve_matching_names` function for more
        information on the name matching.

        Args:
            name_keys: A regular expression or a list of regular expressions to match the body names.
            preserve_order: Whether to preserve the order of the name keys in the output. Defaults to False.

        Returns:
            A tuple of lists containing the body mask, names and indices.
        """
        return find_bodies(self.body_names, name_keys, preserve_order, self.device)

    """
    Operations - Write to simulation.
    """

    @deprecated("write_root_link_pose_to_sim", "write_root_com_velocity_to_sim", since="3.0.0", remove_in="4.0.0")
    def write_root_state_to_sim(
        self,
        root_state: wp.array | torch.Tensor,
        env_ids: Sequence[int] | None = None,
        env_mask: wp.array | torch.Tensor | None = None,
    ) -> None:
        """Set the root state over selected environment indices into the simulation.

        The root state comprises of the cartesian position, quaternion orientation in (x, y, z, w), and angular
        and linear velocity. All the quantities are in the simulation frame.

        Args:
            root_state: Root state in simulation frame. Shape is (len(env_ids), 13) or (num_instances, 13).
            env_ids: Environment indices. If None, then all indices are used.
            env_mask: Environment mask. Shape is (num_instances,).
        """
        # Resolve indices into mask, convert from partial data to complete data, handles the conversion to warp.
        if isinstance(root_state, torch.Tensor):
            root_state = make_complete_data_from_torch_single_index(
                root_state, self.num_instances, ids=env_ids, dtype=vec13f, device=self.device
            )
        env_mask = make_masks_from_torch_ids(self.num_instances, env_ids, env_mask, device=self.device)
        # solve for None masks
        if env_mask is None:
            env_mask = self._data.ALL_ENV_MASK
        # split the state into pose and velocity
        pose, velocity = self._split_state(root_state)
        # write the pose and velocity to the simulation
        self.write_root_link_pose_to_sim(pose, env_mask=env_mask)
        self.write_root_com_velocity_to_sim(velocity, env_mask=env_mask)

    @deprecated("write_root_com_state_to_sim", "write_root_com_velocity_to_sim", since="3.0.0", remove_in="4.0.0")
    def write_root_com_state_to_sim(
        self,
        root_state: wp.array | torch.Tensor,
        env_ids: Sequence[int] | None = None,
        env_mask: wp.array | torch.Tensor | None = None,
    ) -> None:
        """Set the root center of mass state over selected environment indices into the simulation.

        The root state comprises of the cartesian position, quaternion orientation in (x, y, z, w), and angular
        and linear velocity. All the quantities are in the simulation frame.

        Args:
            root_state: Root state in simulation frame. Shape is (len(env_ids), 13) or (num_instances, 13).
            env_ids: Environment indices. If None, then all indices are used.
            env_mask: Environment mask. Shape is (num_instances,).
        """
        # Resolve indices into mask, convert from partial data to complete data, handles the conversion to warp.
        if isinstance(root_state, torch.Tensor):
            root_state = make_complete_data_from_torch_single_index(
                root_state, self.num_instances, ids=env_ids, dtype=vec13f, device=self.device
            )
        env_mask = make_masks_from_torch_ids(self.num_instances, env_ids, env_mask, device=self.device)
        if env_mask is None:
            env_mask = self._data.ALL_ENV_MASK
        # split the state into pose and velocity
        pose, velocity = self._split_state(root_state)
        # write the pose and velocity to the simulation
        self.write_root_com_pose_to_sim(pose, env_mask=env_mask)
        self.write_root_com_velocity_to_sim(velocity, env_mask=env_mask)

    @deprecated("write_root_link_pose_to_sim", "write_root_link_velocity_to_sim", since="3.0.0", remove_in="4.0.0")
    def write_root_link_state_to_sim(
        self,
        root_state: wp.array | torch.Tensor,
        env_ids: Sequence[int] | None = None,
        env_mask: wp.array | torch.Tensor | None = None,
    ) -> None:
        """Set the root link state over selected environment indices into the simulation.

        The root state comprises of the cartesian position, quaternion orientation in (x, y, z, w), and angular
        and linear velocity. All the quantities are in the simulation frame.

        Args:
            root_state: Root state in simulation frame. Shape is (len(env_ids), 13) or (num_instances, 13).
            env_ids: Environment indices. If None, then all indices are used.
            env_mask: Environment mask. Shape is (num_instances,).
        """
        # Resolve indices into mask, convert from partial data to complete data, handles the conversion to warp.
        if isinstance(root_state, torch.Tensor):
            root_state = make_complete_data_from_torch_single_index(
                root_state, self.num_instances, ids=env_ids, dtype=vec13f, device=self.device
            )
        env_mask = make_masks_from_torch_ids(self.num_instances, env_ids, env_mask, device=self.device)
        if env_mask is None:
            env_mask = self._data.ALL_ENV_MASK
        # split the state into pose and velocity
        pose, velocity = self._split_state(root_state)
        # write the pose and velocity to the simulation
        self.write_root_link_pose_to_sim(pose, env_mask=env_mask)
        self.write_root_link_velocity_to_sim(velocity, env_mask=env_mask)

    def write_root_pose_to_sim(
        self,
        root_pose: wp.array | torch.Tensor,
        env_ids: Sequence[int] | None = None,
        env_mask: wp.array | torch.Tensor | None = None,
    ) -> None:
        """Set the root pose over selected environment indices into the simulation.

        The root pose comprises of the cartesian position and quaternion orientation in (x, y, z, w).

        Args:
            root_pose: Root poses in simulation frame. Shape is (len(env_ids), 7) or (num_instances, 7).
            env_ids: Environment indices. If None, then all indices are used.
            env_mask: Environment mask. Shape is (num_instances,).
        """
        self.write_root_link_pose_to_sim(root_pose, env_ids, env_mask)

    def write_root_link_pose_to_sim(
        self,
        pose: wp.array | torch.Tensor,
        env_ids: Sequence[int] | None = None,
        env_mask: wp.array | torch.Tensor | None = None,
    ) -> None:
        """Set the root link pose over selected environment indices into the simulation.


        The root pose ``wp.transformf`` comprises of the cartesian position and quaternion orientation in (x, y, z, w).

        Args:
            root_pose: Root poses in simulation frame. Shape is (len(env_ids), 7) or (num_instances, 7).
            env_ids: Environment indices. If None, then all indices are used.
            env_mask: Environment mask. Shape is (num_instances,).
        """
        # Resolve indices into mask, convert from partial data to complete data, handles the conversion to warp.
        if isinstance(pose, torch.Tensor):
            pose = make_complete_data_from_torch_single_index(
                pose, self.num_instances, ids=env_ids, dtype=wp.transformf, device=self.device
            )
        env_mask = make_masks_from_torch_ids(self.num_instances, env_ids, env_mask, device=self.device)
        # solve for None masks
        if env_mask is None:
            env_mask = self._data.ALL_ENV_MASK
        # set into simulation
        print(f"Pose: {wp.to_torch(pose)}")
        self._update_array_with_array_masked(pose, self._data.root_link_pose_w, env_mask, self.num_instances)
        print(f"Root link pose: {wp.to_torch(self._data.root_link_pose_w)}")
        # invalidate the root com pose
        self._data._root_com_pose_w.timestamp = -1.0

    def write_root_com_pose_to_sim(
        self,
        root_pose: wp.array | torch.Tensor,
        env_ids: Sequence[int] | None = None,
        env_mask: wp.array | torch.Tensor | None = None,
    ) -> None:
        """Set the root center of mass pose over selected environment indices into the simulation.

        The root pose comprises of the cartesian position and quaternion orientation in (x, y, z, w).
        The orientation is the orientation of the principle axes of inertia.

        Args:
            root_pose: Root center of mass poses in simulation frame. Shape is (len(env_ids), 7) or (num_instances, 7).
            env_ids: Environment indices. If None, then all indices are used.
            env_mask: Environment mask. Shape is (num_instances,).
        """
        # Resolve indices into mask, convert from partial data to complete data, handles the conversion to warp.
        if isinstance(root_pose, torch.Tensor):
            root_pose = make_complete_data_from_torch_single_index(
                root_pose, self.num_instances, ids=env_ids, dtype=wp.transformf, device=self.device
            )
        env_mask = make_masks_from_torch_ids(self.num_instances, env_ids, env_mask, device=self.device)
        # solve for None masks
        if env_mask is None:
            env_mask = self._data.ALL_ENV_MASK
        # Write to Newton using warp
        self._update_array_with_array_masked(root_pose, self._data._root_com_pose_w.data, env_mask, self.num_instances)
        # set link frame poses
        wp.launch(
            transform_CoM_pose_to_link_frame_masked_root,
            dim=self.num_instances,
            device=self.device,
            inputs=[
                self._data._root_com_pose_w.data,
                self._data.body_com_pos_b,
                self._data.root_link_pose_w,
                env_mask,
            ],
        )
        # Force update the timestamp
        self._data._root_com_pose_w.timestamp = self._data._sim_timestamp

    def write_root_velocity_to_sim(
        self,
        root_velocity: wp.array | torch.Tensor,
        env_ids: Sequence[int] | None = None,
        env_mask: wp.array | torch.Tensor | None = None,
    ) -> None:
        """Set the root center of mass velocity over selected environment indices into the simulation.

        The velocity comprises linear velocity (x, y, z) and angular velocity (x, y, z) in that order.
        NOTE: This sets the velocity of the root's center of mass rather than the roots frame.

        Args:
            root_velocity: Root center of mass velocities in simulation world frame. Shape is (len(env_ids), 6) or (num_instances, 6).
            env_ids: Environment indices. If None, then all indices are used.
            env_mask: Environment mask. Shape is (num_instances,).
        """
        self.write_root_com_velocity_to_sim(root_velocity, env_ids, env_mask)

    def write_root_com_velocity_to_sim(
        self,
        root_velocity: wp.array | torch.Tensor,
        env_ids: Sequence[int] | None = None,
        env_mask: wp.array | torch.Tensor | None = None,
    ) -> None:
        """Set the root center of mass velocity over selected environment indices into the simulation.

        The velocity comprises linear velocity (x, y, z) and angular velocity (x, y, z) in that order.
        NOTE: This sets the velocity of the root's center of mass rather than the roots frame.

        Args:
            root_velocity: Root center of mass velocities in simulation world frame. Shape is (len(env_ids), 6) or (num_instances, 6).
            env_ids: Environment indices. If None, then all indices are used.
            env_mask: Environment mask. Shape is (num_instances,).
        """
        # Resolve indices into mask, convert from partial data to complete data, handles the conversion to warp.
        if isinstance(root_velocity, torch.Tensor):
            root_velocity = make_complete_data_from_torch_single_index(
                root_velocity, self.num_instances, ids=env_ids, dtype=wp.spatial_vectorf, device=self.device
            )
        env_mask = make_masks_from_torch_ids(self.num_instances, env_ids, env_mask, device=self.device)
        # solve for None masks
        if env_mask is None:
            env_mask = self._data.ALL_ENV_MASK
        # set into simulation
        self._update_array_with_array_masked(root_velocity, self._data.root_com_vel_w, env_mask, self.num_instances)
        # invalidate the derived velocities
        self._data._root_link_vel_w.timestamp = -1.0
        self._data._root_link_vel_b.timestamp = -1.0
        self._data._root_com_vel_b.timestamp = -1.0

    def write_root_link_velocity_to_sim(
        self, root_velocity: wp.array, env_ids: Sequence[int] | None = None, env_mask: wp.array | None = None
    ) -> None:
        """Set the root link velocity over selected environment indices into the simulation.

        The velocity comprises linear velocity (x, y, z) and angular velocity (x, y, z) in that order.
        NOTE: This sets the velocity of the root's frame rather than the roots center of mass.

        Args:
            root_velocity: Root frame velocities in simulation world frame. Shape is (len(env_ids), 6) or (num_instances, 6).
            env_ids: Environment indices. If None, then all indices are used.
            env_mask: Environment mask. Shape is (num_instances,).
        """
        # Resolve indices into mask, convert from partial data to complete data, handles the conversion to warp.
        if isinstance(root_velocity, torch.Tensor):
            root_velocity = make_complete_data_from_torch_single_index(
                root_velocity, self.num_instances, ids=env_ids, dtype=wp.spatial_vectorf, device=self.device
            )
        env_mask = make_masks_from_torch_ids(self.num_instances, env_ids, env_mask, device=self.device)
        # solve for None masks
        if env_mask is None:
            env_mask = self._data.ALL_ENV_MASK
        # update the root link velocity
        self._update_array_with_array_masked(
            root_velocity, self._data._root_link_vel_w.data, env_mask, self.num_instances
        )
        # set into simulation
        wp.launch(
            project_link_velocity_to_com_frame_masked_root,
            dim=self.num_instances,
            device=self.device,
            inputs=[
                root_velocity,
                self._data.root_link_pose_w,
                self._data.body_com_pos_b,
                self._data.root_com_vel_w,
                env_mask,
            ],
        )
        # Force update the timestamp
        self._data._root_link_vel_w.timestamp = self._data._sim_timestamp
        # invalidate the derived velocities
        self._data._root_link_vel_b.timestamp = -1.0
        self._data._root_com_vel_b.timestamp = -1.0

    """
    Operations - Setters.
    """

    def set_masses(
        self,
        masses: torch.Tensor | wp.array,
        body_ids: Sequence[int] | None = None,
        env_ids: Sequence[int] | None = None,
        body_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ):
        """Set masses of all bodies.

        Args:
            masses: Masses of all bodies. Shape is (num_instances, num_bodies).
            body_ids: The body indices to set the masses for. Defaults to None (all bodies).
            env_ids: The environment indices to set the masses for. Defaults to None (all environments).
            body_mask: The body mask. Shape is (num_bodies).
            env_mask: The environment mask. Shape is (num_instances,).
        """
        # raise NotImplementedError()
        if isinstance(masses, torch.Tensor):
            masses = make_complete_data_from_torch_dual_index(
                masses, self.num_instances, self.num_bodies, env_ids, body_ids, dtype=wp.float32, device=self.device
            )
        print(self.num_instances, self.num_bodies)
        print(f"Masses: {wp.to_torch(masses)}")
        env_mask = make_masks_from_torch_ids(self.num_instances, env_ids, env_mask, device=self.device)
        body_mask = make_masks_from_torch_ids(self.num_bodies, body_ids, body_mask, device=self.device)
        print(f"Env mask: {env_mask}")
        print(f"Body mask: {body_mask}")
        print(f"Masses: {wp.to_torch(masses)}")
        # None masks are handled within the kernel.
        self._update_batched_array_with_batched_array_masked(
            masses, self._data.body_mass, env_mask, body_mask, (self.num_instances, self.num_bodies)
        )
        NewtonManager.add_model_change(SolverNotifyFlags.BODY_INERTIAL_PROPERTIES)

    def set_coms(
        self,
        coms: torch.Tensor | wp.array,
        body_ids: Sequence[int] | None = None,
        env_ids: Sequence[int] | None = None,
        body_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ):
        """Set center of mass positions of all bodies.

        Args:
            coms: Center of mass positions of all bodies. Shape is (num_instances, num_bodies, 3).
            body_ids: The body indices to set the center of mass positions for. Defaults to None (all bodies).
            env_ids: The environment indices to set the center of mass positions for. Defaults to None (all environments).
            body_mask: The body mask. Shape is (num_bodies).
            env_mask: The environment mask. Shape is (num_instances,).
        """
        if isinstance(coms, torch.Tensor):
            coms = make_complete_data_from_torch_dual_index(
                coms, self.num_instances, self.num_bodies, env_ids, body_ids, dtype=wp.vec3f, device=self.device
            )
        env_mask = make_masks_from_torch_ids(self.num_instances, env_ids, env_mask, device=self.device)
        body_mask = make_masks_from_torch_ids(self.num_bodies, body_ids, body_mask, device=self.device)
        # None masks are handled within the kernel.
        self._update_batched_array_with_batched_array_masked(
            coms, self._data.body_com_pos_b, env_mask, body_mask, (self.num_instances, self.num_bodies)
        )
        NewtonManager.add_model_change(SolverNotifyFlags.BODY_INERTIAL_PROPERTIES)

    def set_inertias(
        self,
        inertias: torch.Tensor | wp.array,
        body_ids: Sequence[int] | None = None,
        env_ids: Sequence[int] | None = None,
        body_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ):
        """Set inertias of all bodies.

        Args:
            inertias: Inertias of all bodies. Shape is (num_instances, num_bodies, 3, 3).
            body_ids: The body indices to set the inertias for. Defaults to None (all bodies).
            env_ids: The environment indices to set the inertias for. Defaults to None (all environments).
            body_mask: The body mask. Shape is (num_bodies).
            env_mask: The environment mask. Shape is (num_instances,).
        """
        if isinstance(inertias, torch.Tensor):
            inertias = make_complete_data_from_torch_dual_index(
                inertias, self.num_instances, self.num_bodies, env_ids, body_ids, dtype=wp.mat33f, device=self.device
            )
        env_mask = make_masks_from_torch_ids(self.num_instances, env_ids, env_mask, device=self.device)
        body_mask = make_masks_from_torch_ids(self.num_bodies, body_ids, body_mask, device=self.device)
        # None masks are handled within the kernel.
        self._update_batched_array_with_batched_array_masked(
            inertias, self._data.body_inertia, env_mask, body_mask, (self.num_instances, self.num_bodies)
        )
        NewtonManager.add_model_change(SolverNotifyFlags.BODY_INERTIAL_PROPERTIES)

    # TODO: Plug-in the Wrench code from Isaac Lab once the PR gets in.
    def set_external_force_and_torque(
        self,
        forces: torch.Tensor | wp.array,
        torques: torch.Tensor | wp.array,
        body_ids: Sequence[int] | None = None,
        env_ids: Sequence[int] | None = None,
        body_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
        positions: torch.Tensor | wp.array | None = None,
        is_global: bool = False,
    ) -> None:
        """Set external force and torque to apply on the asset's bodies in their local frame.

        For many applications, we want to keep the applied external force on rigid bodies constant over a period of
        time (for instance, during the policy control). This function allows us to store the external force and torque
        into buffers which are then applied to the simulation at every step.

        .. caution::
            If the function is called with empty forces and torques, then this function disables the application
            of external wrench to the simulation.

            .. code-block:: python

                # example of disabling external wrench
                asset.set_external_force_and_torque(forces=wp.zeros(0, 3), torques=wp.zeros(0, 3))

        .. note::
            This function does not apply the external wrench to the simulation. It only fills the buffers with
            the desired values. To apply the external wrench, call the :meth:`write_data_to_sim` function
            right before the simulation step.

        Args:
            forces: External forces in bodies' local frame. Shape is (len(env_ids), len(body_ids), 3) or (num_instances, num_bodies, 3).
            torques: External torques in bodies' local frame. Shape is (len(env_ids), len(body_ids), 3) or (num_instances, num_bodies, 3).
            body_ids: The body indices to set the targets for. Defaults to None (all bodies).
            env_ids: The environment indices to set the targets for. Defaults to None (all environments).
            body_mask: The body mask. Shape is (num_bodies).
            env_mask: The environment mask. Shape is (num_instances,).
            positions: External wrench positions in bodies' local frame. Shape is (len(env_ids), len(body_ids), 3).
                Defaults to None. If None, the external wrench is applied at the center of mass of the body.
            is_global: Whether to apply the external wrench in the global frame. Defaults to False. If set to False,
                the external wrench is applied in the link frame of the articulations' bodies.
        """
        # Resolve indices into mask, convert from partial data to complete data, handles the conversion to warp.
        env_mask_ = None
        body_mask_ = None
        if isinstance(forces, torch.Tensor) or isinstance(torques, torch.Tensor):
            if forces is not None:
                forces = make_complete_data_from_torch_dual_index(
                    forces, self.num_instances, self.num_bodies, env_ids, body_ids, dtype=wp.vec3f, device=self.device
                )
            if torques is not None:
                torques = make_complete_data_from_torch_dual_index(
                    torques, self.num_instances, self.num_bodies, env_ids, body_ids, dtype=wp.vec3f, device=self.device
                )
        env_mask = make_masks_from_torch_ids(self.num_instances, env_ids, env_mask, device=self.device)
        body_mask = make_masks_from_torch_ids(self.num_bodies, body_ids, body_mask, device=self.device)
        # solve for None masks
        if env_mask_ is None:
            env_mask_ = self._data.ALL_ENV_MASK
        if body_mask_ is None:
            body_mask_ = self._data.ALL_BODY_MASK
        # set into simulation
        if (forces is not None) or (torques is not None):
            self.has_external_wrench = True
            if forces is not None:
                self._update_batched_array_with_batched_array_masked(
                    forces, self._external_force_b, env_mask, body_mask, (self.num_instances, self.num_bodies)
                )
            if torques is not None:
                self._update_batched_array_with_batched_array_masked(
                    torques, self._external_torque_b, env_mask, body_mask, (self.num_instances, self.num_bodies)
                )

    """
    Internal helper.
    """

    def _initialize_impl(self):
        # obtain the first prim in the regex expression (all others are assumed to be a copy of this)
        template_prim = sim_utils.find_first_matching_prim(self.cfg.prim_path)
        if template_prim is None:
            raise RuntimeError(f"Failed to find prim for expression: '{self.cfg.prim_path}'.")
        template_prim_path = template_prim.GetPath().pathString

        # find rigid root prims
        root_prims = sim_utils.get_all_matching_child_prims(
            template_prim_path,
            predicate=lambda prim: prim.HasAPI(UsdPhysics.RigidBodyAPI),
            traverse_instance_prims=False,
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
            template_prim_path,
            predicate=lambda prim: prim.HasAPI(UsdPhysics.ArticulationRootAPI),
            traverse_instance_prims=False,
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
        prim_path = root_prim_path_expr.replace(".*", "*")

        self._root_view = NewtonArticulationView(NewtonManager.get_model(), prim_path, verbose=True)

        # container for data access
        self._data = RigidObjectData(self._root_view, self.device)

        # log information about the rigid body
        logger.info(f"Rigid body initialized at: {self.cfg.prim_path} with root '{prim_path}'.")
        logger.info(f"Number of instances: {self.num_instances}")
        logger.info(f"Number of bodies: {self.num_bodies}")
        logger.info(f"Body names: {self.body_names}")

        # process configuration
        self._create_buffers()
        self._process_cfg()
        # update the robot data
        self.update(0.0)
        # Let the articulation data know that it is fully instantiated and ready to use.
        self._data.is_primed = True

    def _create_buffers(self):
        self._ALL_INDICES = torch.arange(self.num_instances, dtype=torch.long, device=self.device)
        self.has_external_wrench = False
        self._external_force_b = wp.zeros((self.num_instances, self.num_bodies), dtype=wp.vec3f, device=self.device)
        self._external_torque_b = wp.zeros((self.num_instances, self.num_bodies), dtype=wp.vec3f, device=self.device)

        # Assign body names to the data
        self._data.body_names = self.body_names

    def _process_cfg(self) -> None:
        """Post processing of configuration parameters."""
        # default pose with quaternion already in (x, y, z, w) format
        default_root_pose = tuple(self.cfg.init_state.pos) + tuple(self.cfg.init_state.rot)
        # update the default root pose
        self._update_array_with_value(
            wp.transformf(*default_root_pose), self._data.default_root_pose, self.num_instances
        )
        # default velocity
        default_root_velocity = tuple(self.cfg.init_state.lin_vel) + tuple(self.cfg.init_state.ang_vel)
        self._update_array_with_value(
            wp.spatial_vectorf(*default_root_velocity), self._data.default_root_vel, self.num_instances
        )

    """
    Internal simulation callbacks.
    """

    def _invalidate_initialize_callback(self, event) -> None:
        """Invalidates the scene elements."""
        # call parent
        super()._invalidate_initialize_callback(event)

    """
    Internal Warp helpers.
    """

    def _update_array_with_value(
        self,
        source: float | int | wp.vec2f | wp.vec3f | wp.quatf | wp.transformf | wp.spatial_vectorf,
        target: wp.array,
        dim: int,
    ):
        """Update an array with a value.

        Args:
            source: The source value.
            target: The target array. Shape is (dim,). Must be pre-allocated, is modified in place.
            dim: The dimension of the array.
        """
        wp.launch(
            update_array1D_with_value,
            dim=(dim,),
            inputs=[
                source,
                target,
            ],
            device=self.device,
        )

    def _update_array_with_array_masked(self, source: wp.array, target: wp.array, mask: wp.array, dim: int):
        """Update an array with an array using a mask.

        Args:
            source: The source array. Shape is (dim,).
            target: The target array. Shape is (dim,). Must be pre-allocated, is modified in place.
            mask: The mask to use. Shape is (dim,).
        """
        wp.launch(
            update_array1D_with_array1D_masked,
            dim=(dim,),
            inputs=[
                source,
                target,
                mask,
            ],
            device=self.device,
        )

    def _update_batched_array_with_batched_array_masked(
        self, source: wp.array, target: wp.array, mask_1: wp.array, mask_2: wp.array, dim: tuple[int, int]
    ):
        """Update a batched array with a batched array using a mask.

        Args:
            source: The source array. Shape is (dim[0], dim[1]).
            target: The target array. Shape is (dim[0], dim[1]). Must be pre-allocated, is modified in place.
            mask_1: The mask to use. Shape is (dim[0],).
            mask_2: The mask to use. Shape is (dim[1],).
            dim: The dimension of the arrays.
        """
        wp.launch(
            update_array2D_with_array2D_masked,
            dim=dim,
            inputs=[
                source,
                target,
                mask_1,
                mask_2,
            ],
            device=self.device,
        )

    def _split_state(
        self,
        state: wp.array,
    ) -> tuple[wp.array, wp.array]:
        """Split the state into pose and velocity.

        Args:
            state: State in simulation frame. Shape is (num_instances, 13).

        Returns:
            A tuple of pose and velocity. Shape is (num_instances, 7) and (num_instances, 6) respectively.
        """
        tmp_pose = wp.zeros((self.num_instances,), dtype=wp.transformf, device=self._device)
        tmp_velocity = wp.zeros((self.num_instances,), dtype=wp.spatial_vectorf, device=self._device)

        wp.launch(
            split_state_to_pose_and_velocity,
            dim=self.num_instances,
            inputs=[
                state,
                tmp_pose,
                tmp_velocity,
            ],
            device=self.device,
        )
        return tmp_pose, tmp_velocity
