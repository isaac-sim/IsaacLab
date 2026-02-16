# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
import torch
import warp as wp

from isaaclab.utils.warp.kernels import (
    add_forces_and_torques_at_position_index,
    add_forces_and_torques_at_position_mask,
    reset_wrench_composer_index,
    reset_wrench_composer_mask,
    set_forces_and_torques_at_position_index,
    set_forces_and_torques_at_position_mask,
)

if TYPE_CHECKING:
    from isaaclab.assets import BaseArticulation, BaseRigidObject, BaseRigidObjectCollection


class WrenchComposer:
    def __init__(self, asset: BaseArticulation | BaseRigidObject | BaseRigidObjectCollection) -> None:
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
            raise ValueError(f"Unsupported asset type: {asset.__class__.__name__}")
        self.device = asset.device
        self._asset = asset
        self._active = False

        # Avoid isinstance here due to potential circular import issues; check by attribute presence instead.
        if hasattr(self._asset.data, "body_link_pose_w"):
            self._get_link_pose_fn = lambda a=self._asset: a.data.body_link_pose_w
        else:
            raise ValueError(f"Unsupported asset type: {self._asset.__class__.__name__}")

        # Create buffers
        self._composed_force_b = wp.zeros((self.num_envs, self.num_bodies), dtype=wp.vec3f, device=self.device)
        self._composed_torque_b = wp.zeros((self.num_envs, self.num_bodies), dtype=wp.vec3f, device=self.device)
        self._ALL_ENV_INDICES = wp.array(np.arange(self.num_envs, dtype=np.int32), dtype=wp.int32, device=self.device)
        self._ALL_BODY_INDICES = wp.array(
            np.arange(self.num_bodies, dtype=np.int32), dtype=wp.int32, device=self.device
        )
        self._ALL_ENV_MASK = wp.ones((self.num_envs), dtype=wp.bool, device=self.device)
        self._ALL_BODY_MASK = wp.ones((self.num_bodies), dtype=wp.bool, device=self.device)

        # Temporary buffers for the masks, positions, and forces/torques (reused to avoid allocations)
        self._temp_env_mask_wp = wp.zeros((self.num_envs,), dtype=wp.bool, device=self.device)
        self._temp_body_mask_wp = wp.zeros((self.num_bodies,), dtype=wp.bool, device=self.device)
        self._temp_positions_wp = wp.zeros((self.num_envs, self.num_bodies), dtype=wp.vec3f, device=self.device)
        self._temp_forces_wp = wp.zeros((self.num_envs, self.num_bodies), dtype=wp.vec3f, device=self.device)
        self._temp_torques_wp = wp.zeros((self.num_envs, self.num_bodies), dtype=wp.vec3f, device=self.device)

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

    def add_forces_and_torques_index(
        self,
        forces: wp.array | torch.Tensor | None = None,
        torques: wp.array | torch.Tensor | None = None,
        positions: wp.array | torch.Tensor | None = None,
        body_ids: torch.Tensor | None = None,
        env_ids: torch.Tensor | None = None,
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
            forces: Forces. (len(env_ids), len(body_ids), 3). Defaults to None.
            torques: Torques. (len(env_ids), len(body_ids), 3). Defaults to None.
            positions: Positions. (len(env_ids), len(body_ids), 3). Defaults to None.
            body_ids: Body ids. Defaults to None (all bodies).
            env_ids: Environment ids. Defaults to None (all environments).
            is_global: Whether the forces and torques are applied in the global frame. Defaults to False.

        Raises:
            ValueError: If the type of the input is not supported.
            ValueError: If the input is a slice and it is not None.
        """
        # Resolve all indices
        if (env_ids is None) or (env_ids == slice(None)):
            env_ids = self._ALL_ENV_INDICES
        if isinstance(env_ids, list):
            env_ids = wp.array(env_ids, dtype=wp.int32, device=self.device)
        if (body_ids is None) or (body_ids == slice(None)):
            body_ids = self._ALL_BODY_INDICES
        if isinstance(body_ids, list):
            body_ids = wp.array(body_ids, dtype=wp.int32, device=self.device)
        if forces is None and torques is None:
            warnings.warn(
                "No forces or torques provided. No force will be added.",
                UserWarning,
                stacklevel=2,
            )
            return
        # Get the link poses
        if not self._link_poses_updated:
            self._link_poses = self._get_link_pose_fn()
            self._link_poses_updated = True

        # Set the active flag to true
        self._active = True

        wp.launch(
            add_forces_and_torques_at_position_index,
            dim=(env_ids.shape[0], body_ids.shape[0]),
            inputs=[
                env_ids,
                body_ids,
                forces,
                torques,
                positions,
                self._link_poses,
                is_global,
            ],
            outputs=[
                self._composed_force_b,
                self._composed_torque_b,
            ],
            device=self.device,
        )

    def set_forces_and_torques_index(
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
        if (env_ids is None) or (env_ids == slice(None)):
            env_ids = self._ALL_ENV_INDICES
        if isinstance(env_ids, list):
            env_ids = wp.array(env_ids, dtype=wp.int32, device=self.device)
        if (body_ids is None) or (body_ids == slice(None)):
            body_ids = self._ALL_BODY_INDICES
        if isinstance(body_ids, list):
            body_ids = wp.array(body_ids, dtype=wp.int32, device=self.device)
        if forces is None and torques is None:
            warnings.warn(
                "No forces or torques provided. No force will be added.",
                UserWarning,
                stacklevel=2,
            )
            return
        # Get the link poses
        if not self._link_poses_updated:
            self._link_poses = self._get_link_pose_fn()
            self._link_poses_updated = True

        # Set the active flag to true
        self._active = True

        wp.launch(
            set_forces_and_torques_at_position_index,
            dim=(env_ids.shape[0], body_ids.shape[0]),
            inputs=[
                env_ids,
                body_ids,
                forces,
                torques,
                positions,
                self._link_poses,
                is_global,
            ],
            outputs=[
                self._composed_force_b,
                self._composed_torque_b,
            ],
            device=self.device,
        )

    def add_forces_and_torques_mask(
        self,
        forces: wp.array | torch.Tensor | None = None,
        torques: wp.array | torch.Tensor | None = None,
        positions: wp.array | torch.Tensor | None = None,
        body_mask: wp.array | torch.Tensor | None = None,
        env_mask: wp.array | torch.Tensor | None = None,
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
            body_mask: Body mask. (num_bodies). Defaults to None (all bodies).
            env_mask: Environment mask. (num_envs). Defaults to None (all environments).
            is_global: Whether the forces and torques are applied in the global frame. Defaults to False.

        Raises:
            ValueError: If the type of the input is not supported.
            ValueError: If the input is a slice and it is not None.
        """
        # Resolve all indices
        if env_mask is None:
            env_mask = self._ALL_ENV_MASK
        if body_mask is None:
            body_mask = self._ALL_BODY_MASK
        if forces is None and torques is None:
            warnings.warn(
                "No forces or torques provided. No force will be added.",
                UserWarning,
                stacklevel=2,
            )
            return
        # Get the link poses
        if not self._link_poses_updated:
            self._link_poses = self._get_link_pose_fn()
            self._link_poses_updated = True

        # Set the active flag to true
        self._active = True

        wp.launch(
            add_forces_and_torques_at_position_mask,
            dim=(self.num_envs, self.num_bodies),
            inputs=[
                env_mask,
                body_mask,
                forces,
                torques,
                positions,
                self._link_poses,
                is_global,
            ],
            outputs=[
                self._composed_force_b,
                self._composed_torque_b,
            ],
            device=self.device,
        )

    def set_forces_and_torques_mask(
        self,
        forces: wp.array | torch.Tensor | None = None,
        torques: wp.array | torch.Tensor | None = None,
        positions: wp.array | torch.Tensor | None = None,
        body_mask: wp.array | torch.Tensor | None = None,
        env_mask: wp.array | torch.Tensor | None = None,
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
            body_mask: Body mask. (num_bodies). Defaults to None (all bodies).
            env_mask: Environment mask. (num_envs). Defaults to None (all environments).
            is_global: Whether the forces and torques are applied in the global frame. Defaults to False.

        Raises:
            ValueError: If the type of the input is not supported.
            ValueError: If the input is a slice and it is not None.
        """
        # Resolve all indices
        if env_mask is None:
            env_mask = self._ALL_ENV_MASK
        if body_mask is None:
            body_mask = self._ALL_BODY_MASK
        if forces is None and torques is None:
            warnings.warn(
                "No forces or torques provided. No force will be added.",
                UserWarning,
                stacklevel=2,
            )
            return
        # Get the link poses
        if not self._link_poses_updated:
            self._link_poses = self._get_link_pose_fn()
            self._link_poses_updated = True

        # Set the active flag to true
        self._active = True

        wp.launch(
            set_forces_and_torques_at_position_mask,
            dim=(self.num_envs, self.num_bodies),
            inputs=[
                env_mask,
                body_mask,
                forces,
                torques,
                positions,
                self._link_poses,
                is_global,
            ],
            outputs=[
                self._composed_force_b,
                self._composed_torque_b,
            ],
            device=self.device,
        )

    def reset(self, env_ids: wp.array | torch.Tensor | None = None, env_mask: wp.array | None = None):
        """Reset the composed force and torque.

        This function will reset the composed force and torque to zero.
        It will also make sure the link positions and quaternions are updated in the next call of the
        `add_forces_and_torques` or `set_forces_and_torques` functions.

        .. note:: This function should be called after every simulation step / reset to ensure no force is carried
        over to the next step.

        .. caution:: If both :attr:`env_ids` and :attr:`env_mask` are provided, then :attr:`env_mask` takes precedence
        over :attr:`env_ids`.

        Args:
            env_ids: Environment indices. Defaults to None (all environments).
            env_mask: Environment mask. Defaults to None (all environments).
        """
        if env_ids is None and env_mask is None:
            self._composed_force_b.zero_()
            self._composed_torque_b.zero_()
            self._active = False
        elif env_mask is not None:
            wp.launch(
                reset_wrench_composer_mask,
                dim=(self.num_envs, self.num_bodies),
                inputs=[
                    env_mask,
                ],
                outputs=[
                    self._composed_force_b,
                    self._composed_torque_b,
                ],
                device=self.device,
            )
        else:
            if env_ids is None or env_ids == slice(None):
                env_ids = self._ALL_ENV_INDICES
            elif isinstance(env_ids, list):
                env_ids = wp.array(env_ids, dtype=wp.int32, device=self.device)
            elif isinstance(env_ids, torch.Tensor):
                env_ids = wp.from_torch(env_ids.to(torch.int32), dtype=wp.int32)
            wp.launch(
                reset_wrench_composer_index,
                dim=(env_ids.shape[0], self.num_bodies),
                inputs=[
                    env_ids,
                ],
                outputs=[
                    self._composed_force_b,
                    self._composed_torque_b,
                ],
                device=self.device,
            )
        self._link_poses_updated = False

    """
    Deprecated functions.
    """

    def add_forces_and_torques(
        self,
        forces: wp.array | torch.Tensor | None = None,
        torques: wp.array | torch.Tensor | None = None,
        positions: wp.array | torch.Tensor | None = None,
        body_ids: torch.Tensor | None = None,
        env_ids: torch.Tensor | None = None,
        is_global: bool = False,
    ):
        """Deprecated, same as :meth:`add_forces_and_torques_index`."""
        warnings.warn(
            "The function 'add_forces_and_torques' will be deprecated in a future release. Please"
            " use 'add_forces_and_torques_index' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.add_forces_and_torques_index(forces, torques, positions, body_ids, env_ids, is_global)

    def set_forces_and_torques(
        self,
        forces: wp.array | torch.Tensor | None = None,
        torques: wp.array | torch.Tensor | None = None,
        positions: wp.array | torch.Tensor | None = None,
        body_ids: wp.array | torch.Tensor | None = None,
        env_ids: wp.array | torch.Tensor | None = None,
        is_global: bool = False,
    ):
        """Deprecated, same as :meth:`set_forces_and_torques_index`."""
        warnings.warn(
            "The function 'set_forces_and_torques' will be deprecated in a future release. Please"
            " use 'set_forces_and_torques_index' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.set_forces_and_torques_index(forces, torques, positions, body_ids, env_ids, is_global)
