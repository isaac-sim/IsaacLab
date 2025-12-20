# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
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
from isaaclab_newton.kernels import (
    generate_mask_from_ids,
    project_link_velocity_to_com_frame_masked_root,
    split_state_to_pose,
    split_state_to_velocity,
    transform_CoM_pose_to_link_frame_masked_root,
    update_wrench_array_with_force,
    update_wrench_array_with_torque,
    vec13f,
)
from newton.selection import ArticulationView as NewtonArticulationView
from newton.solvers import SolverNotifyFlags
from pxr import UsdPhysics

import isaaclab.sim as sim_utils
import isaaclab.utils.string as string_utils
from isaaclab.assets.rigid_object.base_rigid_object import BaseRigidObject
from isaaclab.sim._impl.newton_manager import NewtonManager
from isaaclab.utils.helpers import deprecated
from isaaclab.utils.warp.update_kernels import (
    update_array1D_with_array1D_masked,
    update_array2D_with_array2D_masked,
    update_array2D_with_value_masked,
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
    def body_names(self) -> list[str]:
        """Ordered names of bodies in the rigid object."""
        return self._root_view.link_names

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
        wp.launch(
            update_array2D_with_value_masked,
            dim=(self.num_instances, self.num_bodies),
            inputs=[
                wp.spatial_vectorf(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                self._data._sim_bind_body_external_wrench,
                mask,
                self._data.ALL_ENV_MASK,
            ],
        )

    def write_data_to_sim(self) -> None:
        """Write external wrench to the simulation.

        Note:
            We write external wrench to the simulation here since this function is called before the simulation step.
            This ensures that the external wrench is applied at every simulation step.
        """
        pass

    def update(self, dt: float) -> None:
        self._data.update(dt)

    """
    Frontend conversions - Torch to Warp.
    """

    def _torch_to_warp_single_index(
        self,
        value: torch.Tensor,
        N: int,
        ids: Sequence[int] | None = None,
        mask: torch.Tensor | None = None,
        dtype: type = wp.float32,
    ) -> tuple[wp.array, wp.array | None]:
        """Converts any Torch frontend data into warp data with single index support.

        Args:
            value: The value to convert. Shape is (N,).
            N: The number of elements in the value.
            ids: The index ids.
            mask: The index mask.

        Returns:
            A tuple of warp data with its mask.
        """
        if mask is None:
            if ids is not None:
                # Create a mask from scratch
                env_mask = torch.zeros(N, dtype=torch.bool, device=self.device)
                env_mask[ids] = True
                env_mask = wp.from_torch(env_mask, dtype=wp.bool)
                # Create a complete data buffer from scratch
                complete = torch.zeros((N, *value.shape[1:]), dtype=value.dtype, device=self.device)
                complete[ids] = value
                value = wp.from_torch(complete, dtype=dtype)
            else:
                value = wp.from_torch(value, dtype=dtype)
        else:
            if ids is not None:
                warnings.warn(
                    "ids is not None, but mask is provided. Ignoring ids. Please make sure you are providing complete"
                    " data buffers.",
                    UserWarning,
                )
            env_mask = wp.from_torch(mask, dtype=wp.bool)
            value = wp.from_torch(value, dtype=dtype)
        return value, env_mask

    def _torch_to_warp_dual_index(
        self,
        value: torch.Tensor,
        N: int,
        M: int,
        first_ids: Sequence[int] | None = None,
        second_ids: Sequence[int] | None = None,
        first_mask: torch.Tensor | None = None,
        second_mask: torch.Tensor | None = None,
        dtype: type = wp.float32,
    ) -> tuple[wp.array, wp.array | None, wp.array | None]:
        """Converts any Torch frontend data into warp data with dual index support.

        Args:
            value: The value to convert. Shape is (N, M) or (len(first_ids), len(second_ids)).
            first_ids: The first index ids.
            second_ids: The second index ids.
            first_mask: The first index mask.
            second_mask: The second index mask.
            dtype: The dtype of the value.

        Returns:
            A tuple of warp data with its two masks.
        """
        if first_mask is None:
            if (first_ids is not None) or (second_ids is not None):
                # If we are provided with either first_ids or second_ids, we need to create a mask from scratch and
                # we are expecting partial values.
                if first_ids is not None:
                    # Create a mask from scratch
                    first_mask = torch.zeros(N, dtype=torch.bool, device=self.device)
                    first_mask[first_ids] = True
                    first_mask = wp.from_torch(first_mask, dtype=wp.bool)
                else:
                    first_ids = slice(None)
                if second_ids is not None:
                    # Create a mask from scratch
                    second_mask = torch.zeros(M, dtype=torch.bool, device=self.device)
                    second_mask[second_ids] = True
                    second_mask = wp.from_torch(second_mask, dtype=wp.bool)
                else:
                    second_ids = slice(None)
                if first_ids != slice(None) and second_ids != slice(None):
                    first_ids = first_ids[:, None]

                # Create a complete data buffer from scratch
                complete_value = torch.zeros(N, M, dtype=value.dtype, device=self.device)
                complete_value[first_ids, second_ids] = value
                value = wp.from_torch(complete_value, dtype=dtype)
            elif second_mask is not None:
                second_mask = wp.from_torch(second_mask, dtype=wp.bool)
                value = wp.from_torch(value, dtype=dtype)
            else:
                value = wp.from_torch(value, dtype=dtype)
        else:
            if (first_ids is not None) or (second_ids is not None):
                warnings.warn(
                    "Mask and ids are provided. Ignoring ids. Please make sure you are providing complete data"
                    " buffers.",
                    UserWarning,
                )
            first_mask = wp.from_torch(first_mask, dtype=wp.bool)
            if second_mask is not None:
                second_mask = wp.from_torch(second_mask, dtype=wp.bool)
            else:
                value = wp.from_torch(value, dtype=dtype)

        return value, first_mask, second_mask

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
        return self._find_bodies(name_keys, preserve_order)

    """
    Operations - Write to simulation.
    """

    @deprecated("write_root_link_pose_to_sim", "write_root_com_velocity_to_sim", since="3.0.0", remove_in="4.0.0")
    def write_root_state_to_sim(
        self,
        root_state: torch.Tensor | wp.array,
        env_ids: Sequence[int] | None = None,
        env_mask: torch.Tensor | wp.array | None = None,
    ) -> None:
        """Set the root state over selected environment indices into the simulation.

        The root state comprises of the cartesian position, quaternion orientation in (x, y, z, w), and linear
        and angular velocity. All the quantities are in the simulation frame.

        ..note:: When ids are provided, then partial data is expected. When masks are provided the whole of the data
        is expected. For example, if env_ids is provided, then root_state should be of shape (len(env_ids), 13). If
        env_mask is provided, then root_state should be of shape (num_instances, 13).

        ..caution:: If both env_mask and env_ids are provided, then env_mask will be used. The function will thus
        expect the whole of the data to be provided. If none of them are provided, then the function expects the whole
        of the data to be provided.

        Args:
            root_state: Root state in simulation frame. Shape is (len(env_ids), 13) or (num_instances, 13).
            env_ids: Environment indices. If None, then all indices are used.
            env_mask: Environment mask. Shape is (num_instances,).
        """
        # Resolve indices into mask, convert from partial data to complete data, handles the conversion to warp.
        if isinstance(root_state, torch.Tensor):
            root_state, env_mask = self._torch_to_warp_single_index(
                root_state, self.num_instances, env_ids, env_mask, dtype=vec13f
            )
        # solve for None masks
        if env_mask is None:
            env_mask = self._data.ALL_ENV_MASK
        # split the state into pose and velocity
        pose, velocity = self._split_state(root_state)
        tmp_pose = wp.zeros((self.num_instances,), dtype=wp.transformf, device=self._device)
        tmp_velocity = wp.zeros((self.num_instances,), dtype=wp.spatial_vectorf, device=self._device)

        wp.launch(
            split_state_to_pose,
            dim=self.num_instances,
            inputs=[
                root_state,
                tmp_pose,
            ],
        )
        wp.launch(
            split_state_to_velocity,
            dim=self.num_instances,
            inputs=[
                root_state,
                tmp_velocity,
            ],
        )
        # write the pose and velocity to the simulation
        self.write_root_link_pose_to_sim(tmp_pose, env_mask=env_mask)
        self.write_root_com_velocity_to_sim(tmp_velocity, env_mask=env_mask)

    @deprecated("write_root_com_state_to_sim", "write_root_com_velocity_to_sim", since="3.0.0", remove_in="4.0.0")
    def write_root_com_state_to_sim(
        self,
        root_state: torch.Tensor | wp.array,
        env_ids: Sequence[int] | None = None,
        env_mask: torch.Tensor | wp.array | None = None,
    ) -> None:
        """Set the root center of mass state over selected environment indices into the simulation.

        The root state comprises of the cartesian position, quaternion orientation in (x, y, z, w), and linear
        and angular velocity. All the quantities are in the simulation frame.

        ..note:: When ids are provided, then partial data is expected. When masks are provided the whole of the data
        is expected. For example, if env_ids is provided, then root_state should be of shape (len(env_ids), 13). If
        env_mask is provided, then root_state should be of shape (num_instances, 13).

        ..caution:: If both env_mask and env_ids are provided, then env_mask will be used. The function will thus
        expect the whole of the data to be provided. If none of them are provided, then the function expects the whole
        of the data to be provided.

        Args:
            root_state: Root state in simulation frame. Shape is (len(env_ids), 13) or (num_instances, 13).
            env_ids: Environment indices. If None, then all indices are used.
            env_mask: Environment mask. Shape is (num_instances,).
        """
        # Resolve indices into mask, convert from partial data to complete data, handles the conversion to warp.
        if isinstance(root_state, torch.Tensor):
            root_state, env_mask = self._torch_to_warp_single_index(
                root_state, self.num_instances, env_ids, env_mask, dtype=vec13f
            )
        if env_mask is None:
            env_mask = self._data.ALL_ENV_MASK
        # split the state into pose and velocity
        tmp_pose = wp.zeros((self.num_instances,), dtype=wp.transformf, device=self._device)
        tmp_velocity = wp.zeros((self.num_instances,), dtype=wp.spatial_vectorf, device=self._device)

        wp.launch(
            split_state_to_pose,
            dim=self.num_instances,
            inputs=[
                root_state,
                tmp_pose,
            ],
        )
        wp.launch(
            split_state_to_velocity,
            dim=self.num_instances,
            inputs=[
                root_state,
                tmp_velocity,
            ],
        )
        # write the pose and velocity to the simulation
        self.write_root_com_pose_to_sim(tmp_pose, env_mask=env_mask)
        self.write_root_com_velocity_to_sim(tmp_velocity, env_mask=env_mask)

    @deprecated("write_root_link_pose_to_sim", "write_root_link_velocity_to_sim", since="3.0.0", remove_in="4.0.0")
    def write_root_link_state_to_sim(
        self,
        root_state: torch.Tensor | wp.array,
        env_ids: Sequence[int] | None = None,
        env_mask: torch.Tensor | wp.array | None = None,
    ) -> None:
        """Set the root link state over selected environment indices into the simulation.

        The root state comprises of the cartesian position, quaternion orientation in (x, y, z, w), and linear
        and angular velocity. All the quantities are in the simulation frame.

        ..note:: When ids are provided, then partial data is expected. When masks are provided the whole of the data
        is expected. For example, if env_ids is provided, then root_state should be of shape (len(env_ids), 13). If
        env_mask is provided, then root_state should be of shape (num_instances, 13).

        ..caution:: If both env_mask and env_ids are provided, then env_mask will be used. The function will thus
        expect the whole of the data to be provided. If none of them are provided, then the function expects the whole
        of the data to be provided.

        Args:
            root_state: Root state in simulation frame. Shape is (len(env_ids), 13) or (num_instances, 13).
            env_mask: Environment mask. Shape is (num_instances,).
            env_ids: Environment indices. If None, then all indices are used.
        """
        # Resolve indices into mask, convert from partial data to complete data, handles the conversion to warp.
        if isinstance(root_state, torch.Tensor):
            root_state, env_mask = self._torch_to_warp_single_index(
                root_state, self.num_instances, env_ids, env_mask, dtype=vec13f
            )
        if env_mask is None:
            env_mask = self._data.ALL_ENV_MASK
        # split the state into pose and velocity
        tmp_pose = wp.zeros((self.num_instances,), dtype=wp.transformf, device=self._device)
        tmp_velocity = wp.zeros((self.num_instances,), dtype=wp.spatial_vectorf, device=self._device)

        wp.launch(
            split_state_to_pose,
            dim=self.num_instances,
            inputs=[
                root_state,
                tmp_pose,
            ],
        )
        wp.launch(
            split_state_to_velocity,
            dim=self.num_instances,
            inputs=[
                root_state,
                tmp_velocity,
            ],
        )
        # write the pose and velocity to the simulation
        self.write_root_link_pose_to_sim(tmp_pose, env_mask=env_mask)
        self.write_root_link_velocity_to_sim(tmp_velocity, env_mask=env_mask)

    def write_root_pose_to_sim(
        self,
        root_pose: torch.Tensor | wp.array,
        env_ids: Sequence[int] | None = None,
        env_mask: torch.Tensor | wp.array | None = None,
    ) -> None:
        """Set the root pose over selected environment indices into the simulation.

        The root pose comprises of the cartesian position and quaternion orientation in (x, y, z, w).

        ..note:: When ids are provided, then partial data is expected. When masks are provided the whole of the data
        is expected. For example, if env_ids is provided, then root_pose should be of shape (len(env_ids), 7). If
        env_mask is provided, then root_pose should be of shape (num_instances, 7).

        ..caution:: If both env_mask and env_ids are provided, then env_mask will be used. The function will thus
        expect the whole of the data to be provided. If none of them are provided, then the function expects the whole
        of the data to be provided.

        Args:
            root_pose: Root link poses in simulation frame. Shape is (len(env_ids), 7) or (num_instances, 7).
            env_ids: Environment indices. If None, then all indices are used.
            env_mask: Environment mask. Shape is (num_instances,).
        """
        self.write_root_link_pose_to_sim(root_pose, env_ids, env_mask)

    def write_root_link_pose_to_sim(
        self,
        root_pose: torch.Tensor | wp.array,
        env_ids: Sequence[int] | None = None,
        env_mask: torch.Tensor | wp.array | None = None,
    ) -> None:
        """Set the root link pose over selected environment indices into the simulation.

        The root pose comprises of the cartesian position and quaternion orientation in (x, y, z, w).

        ..note:: When ids are provided, then partial data is expected. When masks are provided the whole of the data
        is expected. For example, if env_ids is provided, then root_pose should be of shape (len(env_ids), 7). If
        env_mask is provided, then root_pose should be of shape (num_instances, 7).

        ..caution:: If both env_mask and env_ids are provided, then env_mask will be used. The function will thus
        expect the whole of the data to be provided. If none of them are provided, then the function expects the whole
        of the data to be provided.

        Args:
            root_pose: Root link poses in simulation frame. Shape is (len(env_ids), 7) or (num_instances, 7).
            env_mask: Environment mask. Shape is (num_instances,).
            env_ids: Environment indices. If None, then all indices are used.
        """
        # Resolve indices into mask, convert from partial data to complete data, handles the conversion to warp.
        if isinstance(root_pose, torch.Tensor):
            root_pose, env_mask = self._torch_to_warp_single_index(
                root_pose, self.num_instances, env_ids, env_mask, dtype=wp.transformf
            )
        # solve for None masks
        if env_mask is None:
            env_mask = self._data.ALL_ENV_MASK
        # set into simulation
        wp.launch(
            update_array1D_with_array1D_masked,
            dim=(self.num_instances,),
            inputs=[
                root_pose,
                self._data.root_link_pose_w,
                env_mask,
                env_mask,
            ],
        )
        # invalidate the root com pose
        self._data._root_com_pose_w.timestamp = -1.0

    def write_root_com_pose_to_sim(
        self,
        root_pose: torch.Tensor | wp.array,
        env_ids: Sequence[int] | None = None,
        env_mask: torch.Tensor | wp.array | None = None,
    ) -> None:
        """Set the root center of mass pose over selected environment indices into the simulation.

        The root pose comprises of the cartesian position and quaternion orientation in (x, y, z, w).
        The orientation is the orientation of the principle axes of inertia.

        ..note:: When ids are provided, then partial data is expected. When masks are provided the whole of the data
        is expected. For example, if env_ids is provided, then root_pose should be of shape (len(env_ids), 7). If
        env_mask is provided, then root_pose should be of shape (num_instances, 7).

        ..caution:: If both env_mask and env_ids are provided, then env_mask will be used. The function will thus
        expect the whole of the data to be provided. If none of them are provided, then the function expects the whole
        of the data to be provided.

        Args:
            root_pose: Root center of mass poses in simulation frame. Shape is (len(env_ids), 7) or (num_instances, 7).
            env_mask: Environment mask. Shape is (num_instances,).
            env_ids: Environment indices. If None, then all indices are used.
        """
        # resolve all indices
        if isinstance(root_pose, torch.Tensor):
            root_pose, env_mask = self._torch_to_warp_single_index(
                root_pose, self.num_instances, env_ids, env_mask, dtype=wp.transformf
            )
        # solve for None masks
        if env_mask is None:
            env_mask = self._data.ALL_ENV_MASK
        # Write to Newton using warp
        self._update_array_with_array_masked(root_pose, self._data.root_com_pose_w.data, env_mask, self.num_instances)
        # set link frame poses
        wp.launch(
            transform_CoM_pose_to_link_frame_masked_root,
            dim=self.num_instances,
            inputs=[
                self._data.root_com_pose_w,
                self._data.body_com_pos_b,
                self._data.root_link_pose_w,
                env_mask,
            ],
        )

    def write_root_velocity_to_sim(
        self,
        root_velocity: torch.Tensor | wp.array,
        env_ids: Sequence[int] | None = None,
        env_mask: torch.Tensor | wp.array | None = None,
    ) -> None:
        """Set the root center of mass velocity over selected environment indices into the simulation.

        The velocity comprises linear velocity (x, y, z) and angular velocity (x, y, z) in that order.
        ..note:: This sets the velocity of the root's center of mass rather than the roots frame.

        ..note:: When ids are provided, then partial data is expected. When masks are provided the whole of the data
        is expected. For example, if env_ids is provided, then root_velocity should be of shape (len(env_ids), 6). If
        env_mask is provided, then root_velocity should be of shape (num_instances, 6).

        ..caution:: If both env_mask and env_ids are provided, then env_mask will be used. The function will thus
        expect the whole of the data to be provided. If none of them are provided, then the function expects the whole
        of the data to be provided.

        Args:
            root_velocity: Root center of mass velocities in simulation world frame. Shape is (len(env_ids), 6) or (num_instances, 6).
            env_mask: Environment mask. Shape is (num_instances,).
            env_ids: Environment indices. If None, then all indices are used.
        """
        self.write_root_com_velocity_to_sim(root_velocity, env_ids, env_mask)

    def write_root_com_velocity_to_sim(
        self,
        root_velocity: torch.Tensor | wp.array,
        env_ids: Sequence[int] | None = None,
        env_mask: torch.Tensor | wp.array | None = None,
    ) -> None:
        """Set the root center of mass velocity over selected environment indices into the simulation.

        The velocity comprises linear velocity (x, y, z) and angular velocity (x, y, z) in that order.
        ..note:: This sets the velocity of the root's center of mass rather than the roots frame.

        ..note:: When ids are provided, then partial data is expected. When masks are provided the whole of the data
        is expected. For example, if env_ids is provided, then root_velocity should be of shape (len(env_ids), 6). If
        env_mask is provided, then root_velocity should be of shape (num_instances, 6).

        ..caution:: If both env_mask and env_ids are provided, then env_mask will be used. The function will thus
        expect the whole of the data to be provided. If none of them are provided, then the function expects the whole
        of the data to be provided.

        Args:
            root_velocity: Root center of mass velocities in simulation world frame. Shape is (len(env_ids), 6) or (num_instances, 6).
            env_mask: Environment mask. Shape is (num_instances,).
            env_ids: Environment indices. If None, then all indices are used.
        """
        # Resolve indices into mask, convert from partial data to complete data, handles the conversion to warp.
        if isinstance(root_velocity, torch.Tensor):
            root_velocity, env_mask = self._torch_to_warp_single_index(
                root_velocity, self.num_instances, env_ids, env_mask, dtype=wp.spatial_vectorf
            )
        # solve for None masks
        if env_mask is None:
            env_mask = self._data.ALL_ENV_MASK
        # set into simulation
        wp.launch(
            update_array1D_with_array1D_masked,
            dim=(self.num_instances,),
            inputs=[
                root_velocity,
                self._data.root_com_vel_w,
                env_mask,
            ],
        )
        # invalidate the derived velocities
        self._data._root_link_vel_w.timestamp = -1.0
        self._data._root_link_vel_b.timestamp = -1.0
        self._data._root_com_vel_b.timestamp = -1.0

    def write_root_link_velocity_to_sim(
        self,
        root_velocity: torch.Tensor | wp.array,
        env_ids: Sequence[int] | None = None,
        env_mask: torch.Tensor | wp.array | None = None,
    ) -> None:
        """Set the root link velocity over selected environment indices into the simulation.

        The velocity comprises linear velocity (x, y, z) and angular velocity (x, y, z) in that order.
        ..note:: This sets the velocity of the root's frame rather than the roots center of mass.

        ..note:: When ids are provided, then partial data is expected. When masks are provided the whole of the data
        is expected. For example, if env_ids is provided, then root_velocity should be of shape (len(env_ids), 6). If
        env_mask is provided, then root_velocity should be of shape (num_instances, 6).

        ..caution:: If both env_mask and env_ids are provided, then env_mask will be used. The function will thus
        expect the whole of the data to be provided. If none of them are provided, then the function expects the whole
        of the data to be provided.

        Args:
            root_velocity: Root frame velocities in simulation world frame. Shape is (len(env_ids), 6) or (num_instances, 6).
            env_mask: Environment mask. Shape is (num_instances,).
            env_ids: Environment indices. If None, then all indices are used.
        """
        # Resolve indices into mask, convert from partial data to complete data, handles the conversion to warp.
        if isinstance(root_velocity, torch.Tensor):
            root_velocity, env_mask = self._torch_to_warp_single_index(
                root_velocity, self.num_instances, env_ids, env_mask, dtype=wp.spatial_vectorf
            )
        # solve for None masks
        if env_mask is None:
            env_mask = self._data.ALL_ENV_MASK
        # set into simulation
        wp.launch(
            update_array1D_with_array1D_masked,
            dim=(self.num_instances,),
            inputs=[
                root_velocity,
                self._data._root_link_vel_w.data,
                env_mask,
            ],
        )
        # set into internal buffers
        wp.launch(
            project_link_velocity_to_com_frame_masked_root,
            dim=self.num_instances,
            inputs=[
                root_velocity,
                self._data.root_link_pose_w,
                self._data.body_com_pos_b,
                self._data.root_com_vel_w,
                env_mask,
            ],
        )
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
        """Set masses of all bodies in the simulation world frame.

        Args:
            masses: Masses of all bodies. Shape is (num_instances, num_bodies).
            body_ids: The body indices to set the masses for. Defaults to None (all bodies).
            env_ids: The environment indices to set the masses for. Defaults to None (all environments).
            body_mask: The body mask. Shape is (num_bodies).
            env_mask: The environment mask. Shape is (num_instances,).
        """
        # raise NotImplementedError()
        if isinstance(masses, torch.Tensor):
            masses, env_mask, body_mask = self._torch_to_warp_dual_index(
                masses, self.num_instances, self.num_bodies, env_ids, body_ids, env_mask, body_mask, dtype=wp.float32
            )
        # solve for None masks
        if env_mask is None:
            env_mask = self._data.ALL_ENV_MASK
        if body_mask is None:
            body_mask = self._data.ALL_BODY_MASK

        wp.launch(
            update_array2D_with_array2D_masked,
            dim=(self.num_instances, self.num_bodies),
            inputs=[
                masses,
                self._data.body_mass,
                env_mask,
                body_mask,
            ],
        )
        NewtonManager.add_model_change(SolverNotifyFlags.BODY_PROPERTIES)

    def set_inertias(
        self,
        inertias: torch.Tensor | wp.array,
        body_ids: Sequence[int] | None = None,
        env_ids: Sequence[int] | None = None,
        body_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ):
        """Set inertias of all bodies in the simulation world frame.

        Args:
            inertias: Inertias of all bodies. Shape is (num_instances, num_bodies, 3, 3).
            body_ids: The body indices to set the inertias for. Defaults to None (all bodies).
            env_ids: The environment indices to set the inertias for. Defaults to None (all environments).
            body_mask: The body mask. Shape is (num_bodies).
            env_mask: The environment mask. Shape is (num_instances,).
        """
        if isinstance(inertias, torch.Tensor):
            inertias, env_mask, body_mask = self._torch_to_warp_dual_index(
                inertias, self.num_instances, self.num_bodies, env_ids, body_ids, env_mask, body_mask, dtype=wp.mat33f
            )
        # solve for None masks
        if env_mask is None:
            env_mask = self._data.ALL_ENV_MASK
        if body_mask is None:
            body_mask = self._data.ALL_BODY_MASK
        wp.launch(
            update_array2D_with_array2D_masked,
            dim=(self.num_instances, self.num_bodies),
            inputs=[
                inertias,
                self._data.body_inertia,
                env_mask,
                body_mask,
            ],
        )
        NewtonManager.add_model_change(SolverNotifyFlags.BODY_PROPERTIES)

    def set_external_force_and_torque(
        self,
        forces: torch.Tensor | wp.array,
        torques: torch.Tensor | wp.array,
        body_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | None = None,
        body_mask: torch.Tensor | wp.array | None = None,
        env_mask: torch.Tensor | wp.array | None = None,
    ) -> None:
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
            body_ids: Body indices to apply external wrench to. Defaults to None (all bodies).
            env_ids: Environment indices to apply external wrench to. Defaults to None (all instances).
            body_mask: The body mask. Shape is (num_bodies).
            env_mask: The environment mask. Shape is (num_instances,).
        """
        # Resolve indices into mask, convert from partial data to complete data, handles the conversion to warp.
        if isinstance(forces, torch.Tensor) or isinstance(torques, torch.Tensor):
            if forces is not None:
                forces, env_mask, body_mask = self._torch_to_warp_dual_index(
                    forces,
                    self.num_instances,
                    self.num_bodies,
                    env_ids,
                    body_ids,
                    env_mask,
                    body_mask,
                    dtype=wp.float32,
                )
            if torques is not None:
                torques, env_mask, body_mask = self._torch_to_warp_dual_index(
                    torques,
                    self.num_instances,
                    self.num_bodies,
                    env_ids,
                    body_ids,
                    env_mask,
                    body_mask,
                    dtype=wp.float32,
                )
        # solve for None masks
        if env_mask is None:
            env_mask = self._data.ALL_ENV_MASK
        if body_mask is None:
            body_mask = self._data.ALL_BODY_MASK
        # set into simulation
        if (forces is not None) or (torques is not None):
            self.has_external_wrench = True
            if forces is not None:
                wp.launch(
                    update_wrench_array_with_force,
                    dim=(self.num_instances, self.num_bodies),
                    inputs=[
                        forces,
                        self._data._sim_bind_body_external_wrench,
                        env_mask,
                        body_mask,
                    ],
                )
            if torques is not None:
                wp.launch(
                    update_wrench_array_with_torque,
                    dim=(self.num_instances, self.num_bodies),
                    inputs=[
                        torques,
                        self._data._sim_bind_body_external_wrench,
                        env_mask,
                        body_mask,
                    ],
                )

    """
    Internal helper.
    """

    def _initialize_impl(self):
        # obtain global simulation view
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

        prim_path = template_prim_path.replace(".*", "*")

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

        # Offsets the spawned pose by the default root pose prior to initializing the solver. This ensures that the
        # solver is initialized at the correct pose, avoiding potential miscalculations in the maximum number of
        # constraints or contact required to run the simulation.
        # TODO: Do this is warp directly?
        generated_pose = wp.to_torch(self._data._default_root_pose).clone()
        generated_pose[:, :2] += wp.to_torch(self._root_view.get_root_transforms(NewtonManager.get_model()))[:, :2]
        self._root_view.set_root_transforms(NewtonManager.get_state_0(), generated_pose)
        self._root_view.set_root_transforms(NewtonManager.get_model(), generated_pose)

    def _create_buffers(self):
        self._ALL_INDICES = torch.arange(self.num_instances, dtype=torch.long, device=self.device)

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

    def _find_bodies(
        self, name_keys: str | Sequence[str], preserve_order: bool = False
    ) -> tuple[wp.array, list[str], list[int]]:
        """Find bodies in the articulation based on the name keys.

        Please check the :meth:`isaaclab.utils.string_utils.resolve_matching_names` function for more
        information on the name matching.

        Args:
            name_keys: A regular expression or a list of regular expressions to match the body names.
            preserve_order: Whether to preserve the order of the name keys in the output. Defaults to False.

        Returns:
            A tuple of lists containing the body mask, names, and indices.
        """
        indices, names = string_utils.resolve_matching_names(name_keys, self.body_names, preserve_order)
        self._data.BODY_MASK.fill_(False)
        mask = wp.clone(self._data.BODY_MASK)
        wp.launch(
            generate_mask_from_ids,
            dim=(len(indices),),
            inputs=[
                mask,
                wp.array(indices, dtype=wp.int32, device=self._device),
            ],
        )
        return mask, names, indices
