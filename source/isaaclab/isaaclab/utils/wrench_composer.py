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
from isaaclab.utils.warp.utils import make_complete_data_from_torch_dual_index, resolve_1d_mask

if TYPE_CHECKING:
    from collections.abc import Sequence

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

        # Store references to Tier 1 (sim-bind) buffers for COM pose computation.
        # We intentionally avoid caching body_com_pose_w (a Tier 2 derived property) because
        # it is lazily computed via a Python timestamp guard.  Saving the .data pointer at init
        # time would freeze it at the initial value — subsequent steps would read stale COM
        # world poses since nothing triggers the lazy recomputation.  Instead, we keep the two
        # Tier 1 inputs (body_link_pose_w and body_com_pos_b) and let the wrench kernels
        # compute the COM pose inline.  This is both correct in eager mode and CUDA-graph-
        # capture safe (Tier 1 buffers are stable sim-bind pointers updated by the solver).
        data = self._asset.data
        if hasattr(data, "body_link_pose_w") and hasattr(data, "body_com_pos_b"):
            self._body_link_pose_w = data.body_link_pose_w
            self._body_com_pos_b = data.body_com_pos_b
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

    # ------------------------------------------------------------------
    # Mask resolution (follows ManagerBasedEnvWarp.resolve_env_mask style)
    # ------------------------------------------------------------------

    def _resolve_env_mask(
        self,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_mask: wp.array | torch.Tensor | None = None,
    ) -> wp.array:
        """Resolve environment ids/mask into a warp boolean mask of shape ``(num_envs,)``."""
        return resolve_1d_mask(
            ids=env_ids,
            mask=env_mask,
            all_mask=self._ALL_ENV_MASK_WP,
            scratch_mask=self._temp_env_mask_wp,
            device=self.device,
        )

    def _resolve_body_mask(
        self,
        body_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        body_mask: wp.array | torch.Tensor | None = None,
    ) -> wp.array:
        """Resolve body ids/mask into a warp boolean mask of shape ``(num_bodies,)``."""
        return resolve_1d_mask(
            ids=body_ids,
            mask=body_mask,
            all_mask=self._ALL_BODY_MASK_WP,
            scratch_mask=self._temp_body_mask_wp,
            device=self.device,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

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
        # -- Preparation: resolve every input to a fixed warp array --
        # Mask resolution is unconditional; resolve_1d_mask handles capture guards internally
        # (rejects torch/id paths during capture, allows None→all_mask and wp.array passthrough).
        env_mask = self._resolve_env_mask(env_ids=env_ids, env_mask=env_mask)
        body_mask = self._resolve_body_mask(body_ids=body_ids, body_mask=body_mask)

        if forces is not None and not isinstance(forces, wp.array):
            if wp.get_device().is_capturing:
                raise RuntimeError("WrenchComposer.add_forces_and_torques requires warp arrays during capture.")
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

        if torques is not None and not isinstance(torques, wp.array):
            if wp.get_device().is_capturing:
                raise RuntimeError("WrenchComposer.add_forces_and_torques requires warp arrays during capture.")
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

        if positions is not None and not isinstance(positions, wp.array):
            if wp.get_device().is_capturing:
                raise RuntimeError("WrenchComposer.add_forces_and_torques requires warp arrays during capture.")
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

        # -- Main ops (capturable): all inputs are now resolved warp arrays --
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
                self._body_link_pose_w,
                self._body_com_pos_b,
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

        # -- Preparation: resolve every input to a fixed warp array --
        # Mask resolution is unconditional; resolve_1d_mask handles capture guards internally.
        env_mask = self._resolve_env_mask(env_ids=env_ids, env_mask=env_mask)
        body_mask = self._resolve_body_mask(body_ids=body_ids, body_mask=body_mask)

        if forces is not None and not isinstance(forces, wp.array):
            if wp.get_device().is_capturing:
                raise RuntimeError("WrenchComposer.set_forces_and_torques requires warp arrays during capture.")
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

        if torques is not None and not isinstance(torques, wp.array):
            if wp.get_device().is_capturing:
                raise RuntimeError("WrenchComposer.set_forces_and_torques requires warp arrays during capture.")
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

        if positions is not None and not isinstance(positions, wp.array):
            if wp.get_device().is_capturing:
                raise RuntimeError("WrenchComposer.set_forces_and_torques requires warp arrays during capture.")
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

        # -- Main ops (capturable): all inputs are now resolved warp arrays --
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
                self._body_link_pose_w,
                self._body_com_pos_b,
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
        # -- Preparation: resolve env_mask --
        if env_ids is not None or (env_mask is not None and not isinstance(env_mask, wp.array)):
            if wp.get_device().is_capturing:
                raise RuntimeError(
                    "WrenchComposer.reset requires env_mask(wp.array[bool]) during capture. "
                    "Do not pass env_ids on captured paths."
                )
            env_mask = self._resolve_env_mask(env_ids=env_ids, env_mask=env_mask)

        # -- Main ops (capturable) --
        if env_mask is None:
            self._composed_force_b.zero_()
            self._composed_torque_b.zero_()
            self._active = False
        else:
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
