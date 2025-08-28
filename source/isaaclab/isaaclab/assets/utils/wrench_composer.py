# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import warp as wp

from ..asset_base import AssetBase
from .kernels import add_forces_and_torques_at_position

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

        if (self._asset.__class__.__name__ == "Articulation") or (self._asset.__class__.__name__ == "RigidObject"):
            self._get_com_fn = lambda a=self._asset: a.data.body_com_pose_w[..., :3]
        elif self._asset.__class__.__name__ == "RigidObjectCollection":
            self._get_com_fn = lambda a=self._asset: a.data.object_com_pose_w[..., :3]
        else:
            raise ValueError(f"Unsupported asset type: {self._asset.__class__.__name__}")

        self._com_positions_updated = False

        self._composed_force_b = wp.zeros((num_envs, num_bodies), dtype=wp.vec3f, device=device)
        self._composed_torque_b = wp.zeros((num_envs, num_bodies), dtype=wp.vec3f, device=device)
        self._com_positions = wp.zeros((num_envs, num_bodies), dtype=wp.vec3f, device=device)

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
        return wp.to_torch(self._composed_force_b)

    @property
    def composed_torque_as_torch(self) -> torch.Tensor:
        """Composed torque at the body's center of mass frame as torch tensor.

        .. note:: If some of the torques are applied in the global frame, the composed torque will be in the center
        mass frame of the body.

        Returns:
            torch.Tensor: Composed torque at the body's center of mass frame. (num_envs, num_bodies, 3)
        """
        return wp.to_torch(self._composed_torque_b)

    def add_forces_and_torques(
        self,
        env_ids: wp.array(dtype=wp.int32),
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

        .. note:: Users should call `reset` function after every simulation step to ensure no force is carried over to the next step.

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

        print("is_global", is_global)

        if is_global:
            if not self._com_positions_updated:
                if self._asset is not None:
                    self._com_positions = self._get_com_fn().clone()
                else:
                    raise ValueError(
                        "Center of mass positions are not available. Please provide an asset to the wrench composer."
                    )
                self._com_positions_updated = True

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

    def reset(self):
        """Reset the composed force and torque.

        This function will reset the composed force and torque to zero.
        It will also make sure the center of mass positions are updated in the next call of the `add_forces_and_torques` function.

        .. note:: This function should be called after every simulation step to ensure no force is carried over to the next step.
        """
        self._composed_force_b.zero_()
        self._composed_torque_b.zero_()
        self._com_positions.zero_()
        self._com_positions_updated = False
