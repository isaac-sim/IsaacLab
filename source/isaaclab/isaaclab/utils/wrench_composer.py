# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import warp as wp

from isaaclab.utils.warp.kernels import add_forces_and_torques_at_position, set_forces_and_torques_at_position
from isaaclab.utils.warp.update_kernels import update_array2D_with_value_masked
from isaaclab.utils.warp.utils import make_complete_data_from_torch_dual_index, make_mask_from_torch_ids

if TYPE_CHECKING:
    from isaaclab.assets.articulation.base_articulation import BaseArticulation
    from isaaclab.assets.rigid_object.base_rigid_object import BaseRigidObject


class WrenchComposer:
    def __init__(self, asset: BaseArticulation | BaseRigidObject) -> None:
        """Wrench composer.

        This class is used to compose forces and torques at the body's com frame.
        It can compose global wrenches and local wrenches. The result is always in the com frame of the body.

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
        if hasattr(self._asset.data, "body_com_pose_w"):
            self._com_pose = self._asset.data.body_com_pose_w
        else:
            raise ValueError(f"Unsupported asset type: {self._asset.__class__.__name__}")

        # Create buffers
        self._composed_force_b = wp.zeros((self.num_envs, self.num_bodies), dtype=wp.vec3f, device=self.device)
        self._composed_torque_b = wp.zeros((self.num_envs, self.num_bodies), dtype=wp.vec3f, device=self.device)
        self._ALL_ENV_MASK_WP = wp.ones((self.num_envs,), dtype=wp.bool, device=self.device)
        self._ALL_BODY_MASK_WP = wp.ones((self.num_bodies,), dtype=wp.bool, device=self.device)

        # Temporary buffers for the masks, positions, and forces/torques (reused to avoid allocations)
        self._temp_env_mask_wp = wp.zeros((self.num_envs,), dtype=wp.bool, device=self.device)
        self._temp_body_mask_wp = wp.zeros((self.num_bodies,), dtype=wp.bool, device=self.device)
        self._temp_positions_wp = wp.zeros((self.num_envs, self.num_bodies), dtype=wp.vec3f, device=self.device)
        self._temp_forces_wp = wp.zeros((self.num_envs, self.num_bodies), dtype=wp.vec3f, device=self.device)
        self._temp_torques_wp = wp.zeros((self.num_envs, self.num_bodies), dtype=wp.vec3f, device=self.device)

    @property
    def active(self) -> bool:
        """Whether the wrench composer is active."""
        return self._active

    @property
    def composed_force(self) -> wp.array:
        """Composed force at the body's com frame.

        .. note:: If some of the forces are applied in the global frame, the composed force will be in the com frame
        of the body.

        Returns:
            wp.array: Composed force at the body's com frame. (num_envs, num_bodies, 3)
        """
        return self._composed_force_b

    @property
    def composed_torque(self) -> wp.array:
        """Composed torque at the body's com frame.

        .. note:: If some of the torques are applied in the global frame, the composed torque will be in the com frame
        of the body.

        Returns:
            wp.array: Composed torque at the body's com frame. (num_envs, num_bodies, 3)
        """
        return self._composed_torque_b

    def add_forces_and_torques(
        self,
        forces: wp.array | torch.Tensor | None = None,
        torques: wp.array | torch.Tensor | None = None,
        positions: wp.array | torch.Tensor | None = None,
        body_ids: torch.Tensor | None = None,
        env_ids: torch.Tensor | None = None,
        body_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
        is_global: bool = False,
    ):
        """Add forces and torques to the composed force and torque.

        Composed force and torque are the sum of all the forces and torques applied to the body.
        It can compose global wrenches and local wrenches. The result is always in the com frame of the body.

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
            RuntimeError: If the provided inputs are not supported.
        """
        if isinstance(forces, torch.Tensor) or isinstance(torques, torch.Tensor) or isinstance(positions, torch.Tensor):
            try:
                env_mask = make_mask_from_torch_ids(
                    self.num_envs, env_ids, env_mask, device=self.device, out=self._temp_env_mask_wp
                )
                body_mask = make_mask_from_torch_ids(
                    self.num_bodies, body_ids, body_mask, device=self.device, out=self._temp_body_mask_wp
                )
                if forces is not None:
                    forces = make_complete_data_from_torch_dual_index(
                        forces,
                        self.num_envs,
                        self.num_bodies,
                        env_ids,
                        body_ids,
                        dtype=wp.vec3f,
                        device=self.device,
                        out=self._temp_forces_wp,
                    )
                if torques is not None:
                    torques = make_complete_data_from_torch_dual_index(
                        torques,
                        self.num_envs,
                        self.num_bodies,
                        env_ids,
                        body_ids,
                        dtype=wp.vec3f,
                        device=self.device,
                        out=self._temp_torques_wp,
                    )
                if positions is not None:
                    positions = make_complete_data_from_torch_dual_index(
                        positions,
                        self.num_envs,
                        self.num_bodies,
                        env_ids,
                        body_ids,
                        dtype=wp.vec3f,
                        device=self.device,
                        out=self._temp_positions_wp,
                    )
            except Exception as e:
                raise RuntimeError(
                    f"Provided inputs are not supported: {e}. When using torch tensors, we expect partial data to be"
                    " provided. And all the tensors to come from torch."
                )

        if body_mask is None:
            body_mask = self._ALL_BODY_MASK_WP
        if env_mask is None:
            env_mask = self._ALL_ENV_MASK_WP

        # Set the active flag to true
        self._active = True

        wp.launch(
            add_forces_and_torques_at_position,
            dim=(self.num_envs, self.num_bodies),
            inputs=[
                env_mask,
                body_mask,
                forces,
                torques,
                positions,
                self._com_pose,
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
        body_ids: torch.Tensor | None = None,
        env_ids: torch.Tensor | None = None,
        body_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
        is_global: bool = False,
    ):
        """Set forces and torques to the composed force and torque.

        Composed force and torque are the sum of all the forces and torques applied to the body.
        It can compose global wrenches and local wrenches. The result is always in the com frame of the body.

        The user can provide any combination of forces, torques, and positions.

        Args:
            forces: Forces. (num_envs, num_bodies, 3). Defaults to None.
            torques: Torques. (num_envs, num_bodies, 3). Defaults to None.
            positions: Positions. (num_envs, num_bodies, 3). Defaults to None.
            body_ids: Body ids. (len(body_ids)). Defaults to None (all bodies).
            env_ids: Environment ids. (len(env_ids)). Defaults to None (all environments).
            body_mask: Body mask. (num_bodies). Defaults to None (all bodies).
            env_mask: Environment mask. (num_envs). Defaults to None (all environments).
            is_global: Whether the forces and torques are applied in the global frame. Defaults to False.

        Raises:
            RuntimeError: If the provided inputs are not supported.
        """

        if isinstance(forces, torch.Tensor):
            forces = make_complete_data_from_torch_dual_index(
                forces,
                self.num_envs,
                self.num_bodies,
                env_ids,
                body_ids,
                dtype=wp.vec3f,
                device=self.device,
                out=self._temp_forces_wp,
            )
        if isinstance(torques, torch.Tensor):
            torques = make_complete_data_from_torch_dual_index(
                torques,
                self.num_envs,
                self.num_bodies,
                env_ids,
                body_ids,
                dtype=wp.vec3f,
                device=self.device,
                out=self._temp_torques_wp,
            )
        if isinstance(positions, torch.Tensor):
            positions = make_complete_data_from_torch_dual_index(
                positions,
                self.num_envs,
                self.num_bodies,
                env_ids,
                body_ids,
                dtype=wp.vec3f,
                device=self.device,
                out=self._temp_positions_wp,
            )

        body_mask = make_mask_from_torch_ids(
            self.num_bodies, body_ids, body_mask, device=self.device, out=self._temp_body_mask_wp
        )
        if body_mask is None:
            body_mask = self._ALL_BODY_MASK_WP
        env_mask = make_mask_from_torch_ids(
            self.num_envs, env_ids, env_mask, device=self.device, out=self._temp_env_mask_wp
        )
        if env_mask is None:
            env_mask = self._ALL_ENV_MASK_WP

        # Set the active flag to true
        self._active = True

        wp.launch(
            set_forces_and_torques_at_position,
            dim=(self.num_envs, self.num_bodies),
            inputs=[
                env_mask,
                body_mask,
                forces,
                torques,
                positions,
                self._com_pose,
                self._composed_force_b,
                self._composed_torque_b,
                is_global,
            ],
            device=self.device,
        )

    def reset(self, env_ids: torch.Tensor | None = None, env_mask: wp.array | None = None):
        """Reset the composed force and torque.

        This function will reset the composed force and torque to zero.

        .. note:: This function should be called after every simulation step / reset to ensure no force is carried
        over to the next step.
        """
        if env_ids is None and env_mask is None:
            self._composed_force_b.zero_()
            self._composed_torque_b.zero_()
            self._active = False
        else:
            if env_ids is not None:
                env_mask = make_mask_from_torch_ids(self.num_envs, env_ids, env_mask, device=self.device)

            wp.launch(
                update_array2D_with_value_masked,
                dim=(self.num_envs, self.num_bodies),
                inputs=[
                    wp.vec3f(0.0, 0.0, 0.0),
                    self._composed_force_b,
                    env_mask,
                    self._ALL_BODY_MASK_WP,
                ],
                device=self.device,
            )
            wp.launch(
                update_array2D_with_value_masked,
                dim=(self.num_envs, self.num_bodies),
                inputs=[
                    wp.vec3f(0.0, 0.0, 0.0),
                    self._composed_torque_b,
                    env_mask,
                    self._ALL_BODY_MASK_WP,
                ],
                device=self.device,
            )
