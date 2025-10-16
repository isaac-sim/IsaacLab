# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import warp as wp
import torch

from ..asset_base import AssetBase
from .kernels import add_forces_and_torques_at_position, set_forces_and_torques_at_position

if TYPE_CHECKING:
    import numpy as np
    import torch


class WrenchComposer:
    def __init__(self, num_envs: int, num_bodies: int, device: str, asset: AssetBase | None = None) -> None:
        """Wrench composer.

        This class is used to compose forces and torques at the body's center of mass frame.
        It can compose global wrenches and local wrenches. The result is always in the center of mass frame of the body.

        Args:
            num_envs: Number of environments.
            num_bodies: Number of bodies.
            device: Device to use.
            asset: Asset to use. Defaults to None.
        """
        self.num_envs = num_envs
        self.num_bodies = num_bodies
        self.device = device
        self._asset = asset
        self._active = False

        if (self._asset.__class__.__name__ == "Articulation") or (self._asset.__class__.__name__ == "RigidObject"):
            self._get_com_fn = lambda a=self._asset: a.data.body_com_pos_w[..., :3]
        elif self._asset.__class__.__name__ == "RigidObjectCollection":
            self._get_com_fn = lambda a=self._asset: a.data.object_com_pos_w[..., :3]
        else:
            raise ValueError(f"Unsupported asset type: {self._asset.__class__.__name__}")

        self._com_positions_updated = False

        self._composed_force_b = wp.zeros((num_envs, num_bodies), dtype=wp.vec3f, device=device)
        self._composed_torque_b = wp.zeros((num_envs, num_bodies), dtype=wp.vec3f, device=device)
        self._com_positions = wp.zeros((num_envs, num_bodies), dtype=wp.vec3f, device=device)
        # Pinning the composed force and torque to the torch tensor to avoid copying the data to the torch tensor every time.
        self._composed_force_b_torch = wp.to_torch(self._composed_force_b)
        self._composed_torque_b_torch = wp.to_torch(self._composed_torque_b)
    
    @property
    def active(self) -> bool:
        """Whether the wrench composer is active."""
        return self._active

    @property
    def composed_force(self) -> wp.array:
        """Composed force at the body's center of mass frame.

        .. note:: If some of the forces are applied in the global frame, the composed force will be in the center
        mass frame of the body.

        Returns:
            wp.array: Composed force at the body's center of mass frame. (num_envs, num_bodies, 3)
        """
        return self._composed_force_b

    @property
    def composed_torque(self) -> wp.array:
        """Composed torque at the body's center of mass frame.

        .. note:: If some of the torques are applied in the global frame, the composed torque will be in the center
        mass frame of the body.

        Returns:
            wp.array: Composed torque at the body's center of mass frame. (num_envs, num_bodies, 3)
        """
        return self._composed_torque_b

    @property
    def composed_force_as_numpy(self) -> np.ndarray:
        """Composed force at the body's center of mass frame as numpy array.

        .. note:: If some of the forces are applied in the global frame, the composed force will be in the center
        mass frame of the body.

        Returns:
            np.ndarray: Composed force at the body's center of mass frame. (num_envs, num_bodies, 3)
        """
        return self._composed_force_b.numpy()

    @property
    def composed_torque_as_numpy(self) -> np.ndarray:
        """Composed torque at the body's center of mass frame as numpy array.

        .. note:: If some of the torques are applied in the global frame, the composed torque will be in the center
        mass frame of the body.

        Returns:
            np.ndarray: Composed torque at the body's center of mass frame. (num_envs, num_bodies, 3)
        """
        return self._composed_torque_b.numpy()

    @property
    def composed_force_as_torch(self) -> torch.Tensor:
        """Composed force at the body's center of mass frame as torch tensor.

        .. note:: If some of the forces are applied in the global frame, the composed force will be in the center
        mass frame of the body.

        Returns:
            torch.Tensor: Composed force at the body's center of mass frame. (num_envs, num_bodies, 3)
        """
        return self._composed_force_b_torch

    @property
    def composed_torque_as_torch(self) -> torch.Tensor:
        """Composed torque at the body's center of mass frame as torch tensor.

        .. note:: If some of the torques are applied in the global frame, the composed torque will be in the center
        mass frame of the body.

        Returns:
            torch.Tensor: Composed torque at the body's center of mass frame. (num_envs, num_bodies, 3)
        """
        return self._composed_torque_b_torch

    def add_forces_and_torques(
        self,
        env_ids: wp.array,
        body_ids: wp.array,
        forces: wp.array | None = None,
        torques: wp.array | None = None,
        positions: wp.array | None = None,
        is_global: bool = False,
    ):
        """Add forces and torques to the composed force and torque.

        Composed force and torque are the sum of all the forces and torques applied to the body.
        It can compose global wrenches and local wrenches. The result is always in the center of mass frame of the body.

        The user can provide any combination of forces, torques, and positions.

        .. note:: Users may want to call `reset` function after every simulation step to ensure no force is carried over to the next step.
        However, this may not necessary if the user calls `set_forces_and_torques` function instead of `add_forces_and_torques`.

        Args:
            env_mask: Environment ids. (num_envs)
            body_mask: Body ids. (num_envs, num_bodies)
            forces: Forces. (num_envs, num_bodies, 3)
            torques: Torques. (num_envs, num_bodies, 3)
            positions: Positions. (num_envs, num_bodies, 3)
            is_global: Whether the forces and torques are applied in the global frame.
        """
        if forces is None:
            forces = wp.empty((0, 0), dtype=wp.vec3f, device=self.device)
        if torques is None:
            torques = wp.empty((0, 0), dtype=wp.vec3f, device=self.device)
        if positions is None:
            positions = wp.empty((0, 0), dtype=wp.vec3f, device=self.device)

        if is_global:
            if not self._com_positions_updated:
                if self._asset is not None:
                    self._com_positions = wp.from_torch(self._get_com_fn().clone(), dtype=wp.vec3f)
                else:
                    raise ValueError(
                        "Center of mass positions are not available. Please provide an asset to the wrench composer."
                    )
                self._com_positions_updated = True
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
                self._com_positions,
                self._composed_force_b,
                self._composed_torque_b,
                is_global,
            ],
            device=self.device,
        )

    def set_forces_and_torques(
        self,
        env_ids: wp.array,
        body_ids: wp.array,
        forces: wp.array | None = None,
        torques: wp.array | None = None,
        positions: wp.array | None = None,
        is_global: bool = False,
    ):
        """Set forces and torques to the composed force and torque.

        Composed force and torque are the sum of all the forces and torques applied to the body.
        It can compose global wrenches and local wrenches. The result is always in the center of mass frame of the body.

        The user can provide any combination of forces, torques, and positions.

        Args:
            env_ids: Environment ids. (num_envs)
            body_ids: Body ids. (num_envs, num_bodies)
            forces: Forces. (num_envs, num_bodies, 3)
            torques: Torques. (num_envs, num_bodies, 3)
            positions: Positions. (num_envs, num_bodies, 3)
            is_global: Whether the forces and torques are applied in the global frame.
        """
        if forces is None:
            forces = wp.empty((0, 0), dtype=wp.vec3f, device=self.device)
        if torques is None:
            torques = wp.empty((0, 0), dtype=wp.vec3f, device=self.device)
        if positions is None:
            positions = wp.empty((0, 0), dtype=wp.vec3f, device=self.device)

        if is_global:
            if not self._com_positions_updated:
                if self._asset is not None:
                    self._com_positions = wp.from_torch(self._get_com_fn().clone(), dtype=wp.vec3f)
                else:
                    raise ValueError(
                        "Center of mass positions are not available. Please provide an asset to the wrench composer."
                    )
                self._com_positions_updated = True
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
                self._com_positions,
                self._composed_force_b,
                self._composed_torque_b,
                is_global,
            ],
            device=self.device,
        )

    def reset(self, env_ids: None = None):
        """Reset the composed force and torque.

        This function will reset the composed force and torque to zero.
        It will also make sure the center of mass positions are updated in the next call of the `add_forces_and_torques` function.

        .. note:: This function should be called after every simulation step to ensure no force is carried over to the next step.
        """
        if env_ids is None:
            self._composed_force_b.zero_()
            self._composed_torque_b.zero_()
            self._com_positions.zero_()
            self._com_positions_updated = False
            self._active = False
        elif isinstance(env_ids, slice):
            self._composed_force_b[env_ids].zero_()
            self._composed_torque_b[env_ids].zero_()
            self._com_positions[env_ids].zero_()
            self._com_positions_updated = False
        else:
            env_ids = wp.from_torch(env_ids.to(torch.int32), dtype=wp.int32)
            self._composed_force_b[env_ids].zero_()
            self._composed_torque_b[env_ids].zero_()
            self._com_positions[env_ids].zero_()
            self._com_positions_updated = False
