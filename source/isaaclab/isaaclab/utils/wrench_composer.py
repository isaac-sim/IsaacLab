# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import warp as wp

from isaaclab.utils.math import convert_quat
from isaaclab.utils.warp.kernels import add_forces_and_torques_at_position, set_forces_and_torques_at_position

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject, RigidObjectCollection


class WrenchComposer:
    def __init__(self, asset: Articulation | RigidObject | RigidObjectCollection) -> None:
        """Wrench composer.

        This class is used to compose forces and torques at the body's link frame.
        It can compose global wrenches and local wrenches. The result is always in the link frame of the body.

        Args:
            asset: Asset to use. Defaults to None.
        """
        self.num_envs = asset.num_instances
        # Avoid isinstance to prevent circular import issues, use attribute presence instead.
        if hasattr(asset, "num_bodies"):
            self.num_bodies = asset.num_bodies
        else:
            self.num_bodies = asset.num_objects
        self.device = asset.device
        self._asset = asset
        self._active = False

        # Avoid isinstance here due to potential circular import issues; check by attribute presence instead.
        if hasattr(self._asset.data, "body_link_pos_w") and hasattr(self._asset.data, "body_link_quat_w"):
            self._get_link_position_fn = lambda a=self._asset: a.data.body_link_pos_w[..., :3]
            self._get_link_quaternion_fn = lambda a=self._asset: a.data.body_link_quat_w[..., :4]
        elif hasattr(self._asset.data, "object_link_pos_w") and hasattr(self._asset.data, "object_link_quat_w"):
            self._get_link_position_fn = lambda a=self._asset: a.data.object_link_pos_w[..., :3]
            self._get_link_quaternion_fn = lambda a=self._asset: a.data.object_link_quat_w[..., :4]
        else:
            raise ValueError(f"Unsupported asset type: {self._asset.__class__.__name__}")

        # Create buffers
        self._composed_force_b = wp.zeros((self.num_envs, self.num_bodies), dtype=wp.vec3f, device=self.device)
        self._composed_torque_b = wp.zeros((self.num_envs, self.num_bodies), dtype=wp.vec3f, device=self.device)
        self._ALL_ENV_INDICES_WP = wp.from_torch(
            torch.arange(self.num_envs, dtype=torch.int32, device=self.device), dtype=wp.int32
        )
        self._ALL_BODY_INDICES_WP = wp.from_torch(
            torch.arange(self.num_bodies, dtype=torch.int32, device=self.device), dtype=wp.int32
        )

        # Pinning the composed force and torque to the torch tensor to avoid copying the data to the torch tensor
        self._composed_force_b_torch = wp.to_torch(self._composed_force_b)
        self._composed_torque_b_torch = wp.to_torch(self._composed_torque_b)
        # Pinning the environment and body indices to the torch tensor to allow for slicing.
        self._ALL_ENV_INDICES_TORCH = wp.to_torch(self._ALL_ENV_INDICES_WP)
        self._ALL_BODY_INDICES_TORCH = wp.to_torch(self._ALL_BODY_INDICES_WP)

        # Flag to check if the link poses have been updated.
        self._link_poses_updated = False

    @property
    def active(self) -> bool:
        """Whether the wrench composer is active."""
        return self._active

    @property
    def composed_force(self) -> wp.array:
        """Composed force at the body's link frame.

        .. note:: If some of the forces are applied in the global frame, the composed force will be in the link frame
        of the body.

        Returns:
            wp.array: Composed force at the body's link frame. (num_envs, num_bodies, 3)
        """
        return self._composed_force_b

    @property
    def composed_torque(self) -> wp.array:
        """Composed torque at the body's link frame.

        .. note:: If some of the torques are applied in the global frame, the composed torque will be in the link frame
        of the body.

        Returns:
            wp.array: Composed torque at the body's link frame. (num_envs, num_bodies, 3)
        """
        return self._composed_torque_b

    @property
    def composed_force_as_torch(self) -> torch.Tensor:
        """Composed force at the body's link frame as torch tensor.

        .. note:: If some of the forces are applied in the global frame, the composed force will be in the link frame
        of the body.

        Returns:
            torch.Tensor: Composed force at the body's link frame. (num_envs, num_bodies, 3)
        """
        return self._composed_force_b_torch

    @property
    def composed_torque_as_torch(self) -> torch.Tensor:
        """Composed torque at the body's link frame as torch tensor.

        .. note:: If some of the torques are applied in the global frame, the composed torque will be in the link frame
        of the body.

        Returns:
            torch.Tensor: Composed torque at the body's link frame. (num_envs, num_bodies, 3)
        """
        return self._composed_torque_b_torch

    def add_forces_and_torques(
        self,
        forces: wp.array | torch.Tensor | None = None,
        torques: wp.array | torch.Tensor | None = None,
        positions: wp.array | torch.Tensor | None = None,
        body_ids: wp.array | torch.Tensor | None = None,
        env_ids: wp.array | torch.Tensor | None = None,
        is_global: bool = False,
    ):
        """Add forces and torques to the composed force and torque.

        Composed force and torque are the sum of all the forces and torques applied to the body.
        It can compose global wrenches and local wrenches. The result is always in the link frame of the body.

        The user can provide any combination of forces, torques, and positions.

        .. note:: Users may want to call `reset` function after every simulation step to ensure no force is carried
        over to the next step. However, this may not necessary if the user calls `set_forces_and_torques` function
        instead of `add_forces_and_torques`.

        Args:
            forces: Forces. (num_envs, num_bodies, 3). Defaults to None.
            torques: Torques. (num_envs, num_bodies, 3). Defaults to None.
            positions: Positions. (num_envs, num_bodies, 3). Defaults to None.
            body_ids: Body ids. (num_envs, num_bodies). Defaults to None (all bodies).
            env_ids: Environment ids. (num_envs). Defaults to None (all environments).
            is_global: Whether the forces and torques are applied in the global frame. Defaults to False.

        Raises:
            ValueError: If the type of the input is not supported.
            ValueError: If the input is a slice and it is not None.
        """
        # Resolve all indices
        # -- env_ids
        if env_ids is None:
            env_ids = self._ALL_ENV_INDICES_WP
        elif isinstance(env_ids, torch.Tensor):
            env_ids = wp.from_torch(env_ids.to(torch.int32), dtype=wp.int32)
        elif isinstance(env_ids, list):
            env_ids = wp.array(env_ids, dtype=wp.int32, device=self.device)
        elif isinstance(env_ids, slice):
            if env_ids == slice(None):
                env_ids = self._ALL_ENV_INDICES_WP
            else:
                raise ValueError(f"Doesn't support slice input for env_ids: {env_ids}")
        # -- body_ids
        if body_ids is None:
            body_ids = self._ALL_BODY_INDICES_WP
        elif isinstance(body_ids, torch.Tensor):
            body_ids = wp.from_torch(body_ids.to(torch.int32), dtype=wp.int32)
        elif isinstance(body_ids, list):
            body_ids = wp.array(body_ids, dtype=wp.int32, device=self.device)
        elif isinstance(body_ids, slice):
            if body_ids == slice(None):
                body_ids = self._ALL_BODY_INDICES_WP
            else:
                raise ValueError(f"Doesn't support slice input for body_ids: {body_ids}")

        # Resolve remaining inputs
        # -- don't launch if no forces or torques are provided
        if forces is None and torques is None:
            return
        if isinstance(forces, torch.Tensor):
            forces = wp.from_torch(forces, dtype=wp.vec3f)
        if isinstance(torques, torch.Tensor):
            torques = wp.from_torch(torques, dtype=wp.vec3f)
        if isinstance(positions, torch.Tensor):
            positions = wp.from_torch(positions, dtype=wp.vec3f)

        # Get the link positions and quaternions
        if not self._link_poses_updated:
            self._link_positions = wp.from_torch(self._get_link_position_fn().clone(), dtype=wp.vec3f)
            self._link_quaternions = wp.from_torch(
                convert_quat(self._get_link_quaternion_fn().clone(), to="xyzw"), dtype=wp.quatf
            )
            self._link_poses_updated = True

        # Set the active flag to true
        self._active = True

        wp.launch(
            add_forces_and_torques_at_position,
            dim=(env_ids.shape[0], body_ids.shape[0]),
            inputs=[
                env_ids,
                body_ids,
                forces,
                torques,
                positions,
                self._link_positions,
                self._link_quaternions,
                self._composed_force_b,
                self._composed_torque_b,
                is_global,
            ],
            device=self.device,
        )

    def set_forces_and_torques(
        self,
        forces: wp.array | torch.Tensor | None = None,
        torques: wp.array | torch.Tensor | None = None,
        positions: wp.array | torch.Tensor | None = None,
        body_ids: wp.array | torch.Tensor | None = None,
        env_ids: wp.array | torch.Tensor | None = None,
        is_global: bool = False,
    ):
        """Set forces and torques to the composed force and torque.

        Composed force and torque are the sum of all the forces and torques applied to the body.
        It can compose global wrenches and local wrenches. The result is always in the link frame of the body.

        The user can provide any combination of forces, torques, and positions.

        Args:
            forces: Forces. (num_envs, num_bodies, 3). Defaults to None.
            torques: Torques. (num_envs, num_bodies, 3). Defaults to None.
            positions: Positions. (num_envs, num_bodies, 3). Defaults to None.
            body_ids: Body ids. (num_envs, num_bodies). Defaults to None (all bodies).
            env_ids: Environment ids. (num_envs). Defaults to None (all environments).
            is_global: Whether the forces and torques are applied in the global frame. Defaults to False.

        Raises:
            ValueError: If the type of the input is not supported.
            ValueError: If the input is a slice and it is not None.
        """
        # Resolve all indices
        # -- env_ids
        if env_ids is None:
            env_ids = self._ALL_ENV_INDICES_WP
        elif isinstance(env_ids, torch.Tensor):
            env_ids = wp.from_torch(env_ids.to(torch.int32), dtype=wp.int32)
        elif isinstance(env_ids, list):
            env_ids = wp.array(env_ids, dtype=wp.int32, device=self.device)
        elif isinstance(env_ids, slice):
            if env_ids == slice(None):
                env_ids = self._ALL_ENV_INDICES_WP
            else:
                raise ValueError(f"Doesn't support slice input for env_ids: {env_ids}")
        # -- body_ids
        if body_ids is None:
            body_ids = self._ALL_BODY_INDICES_WP
        elif isinstance(body_ids, torch.Tensor):
            body_ids = wp.from_torch(body_ids.to(torch.int32), dtype=wp.int32)
        elif isinstance(body_ids, list):
            body_ids = wp.array(body_ids, dtype=wp.int32, device=self.device)
        elif isinstance(body_ids, slice):
            if body_ids == slice(None):
                body_ids = self._ALL_BODY_INDICES_WP
            else:
                raise ValueError(f"Doesn't support slice input for body_ids: {body_ids}")
        # Resolve remaining inputs
        # -- don't launch if no forces or torques are provided
        if forces is None and torques is None:
            return
        if forces is None:
            forces = wp.empty((0, 0), dtype=wp.vec3f, device=self.device)
        elif isinstance(forces, torch.Tensor):
            forces = wp.from_torch(forces, dtype=wp.vec3f)
        if torques is None:
            torques = wp.empty((0, 0), dtype=wp.vec3f, device=self.device)
        elif isinstance(torques, torch.Tensor):
            torques = wp.from_torch(torques, dtype=wp.vec3f)
        if positions is None:
            positions = wp.empty((0, 0), dtype=wp.vec3f, device=self.device)
        elif isinstance(positions, torch.Tensor):
            positions = wp.from_torch(positions, dtype=wp.vec3f)

        # Get the link positions and quaternions
        if not self._link_poses_updated:
            self._link_positions = wp.from_torch(self._get_link_position_fn().clone(), dtype=wp.vec3f)
            self._link_quaternions = wp.from_torch(
                convert_quat(self._get_link_quaternion_fn().clone(), to="xyzw"), dtype=wp.quatf
            )
            self._link_poses_updated = True

        # Set the active flag to true
        self._active = True

        wp.launch(
            set_forces_and_torques_at_position,
            dim=(env_ids.shape[0], body_ids.shape[0]),
            inputs=[
                env_ids,
                body_ids,
                forces,
                torques,
                positions,
                self._link_positions,
                self._link_quaternions,
                self._composed_force_b,
                self._composed_torque_b,
                is_global,
            ],
            device=self.device,
        )

    def reset(self, env_ids: wp.array | torch.Tensor | None = None):
        """Reset the composed force and torque.

        This function will reset the composed force and torque to zero.
        It will also make sure the link positions and quaternions are updated in the next call of the
        `add_forces_and_torques` or `set_forces_and_torques` functions.

        .. note:: This function should be called after every simulation step / reset to ensure no force is carried
        over to the next step.
        """
        if env_ids is None:
            self._composed_force_b.zero_()
            self._composed_torque_b.zero_()
            self._active = False
        else:
            indices = env_ids
            if isinstance(env_ids, torch.Tensor):
                indices = wp.from_torch(env_ids.to(torch.int32), dtype=wp.int32)
            elif isinstance(env_ids, list):
                indices = wp.array(env_ids, dtype=wp.int32, device=self.device)
            elif isinstance(env_ids, slice):
                if env_ids == slice(None):
                    indices = self._ALL_ENV_INDICES_WP
                else:
                    indices = env_ids

            self._composed_force_b[indices].zero_()
            self._composed_torque_b[indices].zero_()

        self._link_poses_updated = False
