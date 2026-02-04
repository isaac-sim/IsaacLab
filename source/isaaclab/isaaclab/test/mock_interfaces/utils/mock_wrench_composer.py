# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Mock WrenchComposer for testing and benchmarking.

This module provides a mock implementation of the WrenchComposer class that can be used
in testing and benchmarking without requiring the full simulation environment.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import warp as wp

if TYPE_CHECKING:
    from isaaclab.assets import BaseArticulation, BaseRigidObject, BaseRigidObjectCollection


class MockWrenchComposer:
    """Mock WrenchComposer for testing.

    This class provides a mock implementation of WrenchComposer that matches the real interface
    but does not launch Warp kernels. It can be used for testing and benchmarking asset classes
    without requiring the full simulation environment.

    The mock maintains simple buffers and sets the active flag when forces/torques are added,
    but does not perform actual force composition computations.
    """

    def __init__(self, asset: BaseArticulation | BaseRigidObject | BaseRigidObjectCollection) -> None:
        """Initialize the mock wrench composer.

        Args:
            asset: Asset to use (Articulation, RigidObject, or RigidObjectCollection).
        """
        self.num_envs = asset.num_instances
        if hasattr(asset, "num_bodies"):
            self.num_bodies = asset.num_bodies
        else:
            raise ValueError(f"Unsupported asset type: {asset.__class__.__name__}")
        self.device = asset.device
        self._asset = asset
        self._active = False

        # Create buffers using Warp (matching real WrenchComposer)
        self._composed_force_b = wp.zeros((self.num_envs, self.num_bodies), dtype=wp.vec3f, device=self.device)
        self._composed_torque_b = wp.zeros((self.num_envs, self.num_bodies), dtype=wp.vec3f, device=self.device)

        # Create torch views (matching real WrenchComposer)
        self._composed_force_b_torch = wp.to_torch(self._composed_force_b)
        self._composed_torque_b_torch = wp.to_torch(self._composed_torque_b)

        # Create index arrays
        self._ALL_ENV_INDICES_WP = wp.from_torch(
            torch.arange(self.num_envs, dtype=torch.int32, device=self.device), dtype=wp.int32
        )
        self._ALL_BODY_INDICES_WP = wp.from_torch(
            torch.arange(self.num_bodies, dtype=torch.int32, device=self.device), dtype=wp.int32
        )
        self._ALL_ENV_INDICES_TORCH = wp.to_torch(self._ALL_ENV_INDICES_WP)
        self._ALL_BODY_INDICES_TORCH = wp.to_torch(self._ALL_BODY_INDICES_WP)

    @property
    def active(self) -> bool:
        """Whether the wrench composer is active."""
        return self._active

    @property
    def composed_force(self) -> wp.array:
        """Composed force at the body's link frame.

        Returns:
            wp.array: Composed force at the body's link frame. (num_envs, num_bodies, 3)
        """
        return self._composed_force_b

    @property
    def composed_torque(self) -> wp.array:
        """Composed torque at the body's link frame.

        Returns:
            wp.array: Composed torque at the body's link frame. (num_envs, num_bodies, 3)
        """
        return self._composed_torque_b

    @property
    def composed_force_as_torch(self) -> torch.Tensor:
        """Composed force at the body's link frame as torch tensor.

        Returns:
            torch.Tensor: Composed force at the body's link frame. (num_envs, num_bodies, 3)
        """
        return self._composed_force_b_torch

    @property
    def composed_torque_as_torch(self) -> torch.Tensor:
        """Composed torque at the body's link frame as torch tensor.

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
    ) -> None:
        """Add forces and torques (mock - just sets active flag).

        Args:
            forces: Forces. (num_envs, num_bodies, 3). Defaults to None.
            torques: Torques. (num_envs, num_bodies, 3). Defaults to None.
            positions: Positions. (num_envs, num_bodies, 3). Defaults to None.
            body_ids: Body ids. Defaults to None (all bodies).
            env_ids: Environment ids. Defaults to None (all environments).
            is_global: Whether the forces and torques are applied in the global frame. Defaults to False.
        """
        if forces is not None or torques is not None:
            self._active = True

    def set_forces_and_torques(
        self,
        forces: wp.array | torch.Tensor | None = None,
        torques: wp.array | torch.Tensor | None = None,
        positions: wp.array | torch.Tensor | None = None,
        body_ids: wp.array | torch.Tensor | None = None,
        env_ids: wp.array | torch.Tensor | None = None,
        is_global: bool = False,
    ) -> None:
        """Set forces and torques (mock - just sets active flag).

        Args:
            forces: Forces. (num_envs, num_bodies, 3). Defaults to None.
            torques: Torques. (num_envs, num_bodies, 3). Defaults to None.
            positions: Positions. (num_envs, num_bodies, 3). Defaults to None.
            body_ids: Body ids. Defaults to None (all bodies).
            env_ids: Environment ids. Defaults to None (all environments).
            is_global: Whether the forces and torques are applied in the global frame. Defaults to False.
        """
        if forces is not None or torques is not None:
            self._active = True

    def reset(self, env_ids: wp.array | torch.Tensor | None = None) -> None:
        """Reset the composed force and torque.

        Args:
            env_ids: Environment ids to reset. Defaults to None (all environments).
        """
        if env_ids is None:
            self._composed_force_b.zero_()
            self._composed_torque_b.zero_()
            self._active = False
        else:
            # For partial reset, just zero the specified environments
            if isinstance(env_ids, torch.Tensor):
                indices = wp.from_torch(env_ids.to(torch.int32), dtype=wp.int32)
            elif isinstance(env_ids, list):
                indices = wp.array(env_ids, dtype=wp.int32, device=self.device)
            else:
                indices = env_ids
            self._composed_force_b[indices].zero_()
            self._composed_torque_b[indices].zero_()
