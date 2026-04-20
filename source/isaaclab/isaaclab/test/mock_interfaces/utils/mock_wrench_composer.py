# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Mock WrenchComposer for testing and benchmarking.

This module provides a mock implementation of the WrenchComposer class that can be used
in testing and benchmarking without requiring the full simulation environment.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import torch
import warp as wp

if TYPE_CHECKING:
    from isaaclab.assets import BaseArticulation, BaseRigidObject, BaseRigidObjectCollection


class MockWrenchComposer:
    """Mock WrenchComposer matching the dual-buffer API for testing.

    This class provides a mock implementation of WrenchComposer that matches the real interface
    but does not launch Warp kernels. It can be used for testing and benchmarking asset classes
    without requiring the full simulation environment.

    The mock maintains the 5 input buffers and 2 output buffers matching the real WrenchComposer,
    and sets the active flag when forces/torques are added. The ``compose_to_body_frame()`` method
    simply copies the local buffers to the output buffers (since mock assets typically use identity
    transforms).
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

        # -- Tracking flags --
        self._active: bool = False
        self._dirty: bool = False

        shape = (self.num_envs, self.num_bodies)

        # -- 5 input buffers --
        self._global_force_w = wp.zeros(shape, dtype=wp.vec3f, device=self.device)
        self._global_torque_w = wp.zeros(shape, dtype=wp.vec3f, device=self.device)
        self._global_force_at_com_w = wp.zeros(shape, dtype=wp.vec3f, device=self.device)
        self._local_force_b = wp.zeros(shape, dtype=wp.vec3f, device=self.device)
        self._local_torque_b = wp.zeros(shape, dtype=wp.vec3f, device=self.device)

        # -- 2 output buffers --
        self._out_force_b = wp.zeros(shape, dtype=wp.vec3f, device=self.device)
        self._out_torque_b = wp.zeros(shape, dtype=wp.vec3f, device=self.device)

        # Create index arrays
        self._ALL_ENV_INDICES_WP = wp.from_torch(
            torch.arange(self.num_envs, dtype=torch.int32, device=self.device), dtype=wp.int32
        )
        self._ALL_BODY_INDICES_WP = wp.from_torch(
            torch.arange(self.num_bodies, dtype=torch.int32, device=self.device), dtype=wp.int32
        )
        self._ALL_ENV_INDICES_TORCH = wp.to_torch(self._ALL_ENV_INDICES_WP)
        self._ALL_BODY_INDICES_TORCH = wp.to_torch(self._ALL_BODY_INDICES_WP)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def active(self) -> bool:
        """Whether any forces or torques have been written since the last full reset."""
        return self._active

    # -- Input buffer accessors (read-only) --

    @property
    def global_force_w(self) -> wp.array:
        """Positional global forces buffer. Shape ``(num_envs, num_bodies)``, dtype ``wp.vec3f``."""
        return self._global_force_w

    @property
    def global_torque_w(self) -> wp.array:
        """Global torques buffer (about world origin). Shape ``(num_envs, num_bodies)``, dtype ``wp.vec3f``."""
        return self._global_torque_w

    @property
    def global_force_at_com_w(self) -> wp.array:
        """Global forces at CoM buffer (no positional torque). Shape ``(num_envs, num_bodies)``, dtype ``wp.vec3f``."""
        return self._global_force_at_com_w

    @property
    def local_force_b(self) -> wp.array:
        """Body-frame forces buffer. Shape ``(num_envs, num_bodies)``, dtype ``wp.vec3f``."""
        return self._local_force_b

    @property
    def local_torque_b(self) -> wp.array:
        """Body-frame torques buffer. Shape ``(num_envs, num_bodies)``, dtype ``wp.vec3f``."""
        return self._local_torque_b

    # -- Output buffer accessors --

    @property
    def out_force_b(self) -> wp.array:
        """Composed force in the body (link) frame. Shape ``(num_envs, num_bodies)``, dtype ``wp.vec3f``.

        Triggers composition from input buffers if dirty.
        """
        self._ensure_composed()
        return self._out_force_b

    @property
    def out_torque_b(self) -> wp.array:
        """Composed torque in the body (link) frame. Shape ``(num_envs, num_bodies)``, dtype ``wp.vec3f``.

        Triggers composition from input buffers if dirty.
        """
        self._ensure_composed()
        return self._out_torque_b

    # -- Legacy composed_force / composed_torque properties for backward compat --

    @property
    def composed_force(self) -> wp.array:
        """Composed force at the body's link frame.

        .. deprecated:: 4.5.33
            Use :attr:`out_force_b` instead.
        """
        warnings.warn(
            "The property 'composed_force' is deprecated. Use 'out_force_b' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.out_force_b

    @property
    def composed_torque(self) -> wp.array:
        """Composed torque at the body's link frame.

        .. deprecated:: 4.5.33
            Use :attr:`out_torque_b` instead.
        """
        warnings.warn(
            "The property 'composed_torque' is deprecated. Use 'out_torque_b' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.out_torque_b

    # ------------------------------------------------------------------
    # Composition
    # ------------------------------------------------------------------

    def compose_to_body_frame(self):
        """Mock composition: sums all input buffers to output assuming identity transforms.

        Under identity transforms (no rotation), global-frame values equal body-frame values,
        so all five input buffers are summed directly into the two output buffers.
        """
        # Zero output buffers
        self._out_force_b.zero_()
        self._out_torque_b.zero_()

        # Use torch views for the accumulation
        out_force_torch = wp.to_torch(self._out_force_b)
        out_torque_torch = wp.to_torch(self._out_torque_b)

        # Sum all force contributions (identity: no rotation needed)
        out_force_torch.add_(wp.to_torch(self._local_force_b))
        out_force_torch.add_(wp.to_torch(self._global_force_w))
        out_force_torch.add_(wp.to_torch(self._global_force_at_com_w))

        # Sum all torque contributions
        out_torque_torch.add_(wp.to_torch(self._local_torque_b))
        out_torque_torch.add_(wp.to_torch(self._global_torque_w))

        self._dirty = False

    # ------------------------------------------------------------------
    # Buffer merging
    # ------------------------------------------------------------------

    def add_raw_buffers_from(self, other: MockWrenchComposer):
        """Element-wise add another composer's five input buffers into this one.

        Args:
            other: Another :class:`MockWrenchComposer` whose input buffers will be added into this one.
        """
        # Use torch views for element-wise addition
        wp.to_torch(self._global_force_w).add_(wp.to_torch(other._global_force_w))
        wp.to_torch(self._global_torque_w).add_(wp.to_torch(other._global_torque_w))
        wp.to_torch(self._global_force_at_com_w).add_(wp.to_torch(other._global_force_at_com_w))
        wp.to_torch(self._local_force_b).add_(wp.to_torch(other._local_force_b))
        wp.to_torch(self._local_torque_b).add_(wp.to_torch(other._local_torque_b))

        if other._active:
            self._active = True
            self._dirty = True

    # ------------------------------------------------------------------
    # Add / Set methods
    # ------------------------------------------------------------------

    def add_forces_and_torques(
        self,
        forces: wp.array | torch.Tensor | None = None,
        torques: wp.array | torch.Tensor | None = None,
        positions: wp.array | torch.Tensor | None = None,
        body_ids: wp.array | torch.Tensor | None = None,
        env_ids: wp.array | torch.Tensor | None = None,
        is_global: bool = False,
    ) -> None:
        """Add forces and torques (deprecated, use add_forces_and_torques_index).

        Args:
            forces: Forces. (num_envs, num_bodies, 3). Defaults to None.
            torques: Torques. (num_envs, num_bodies, 3). Defaults to None.
            positions: Positions. (num_envs, num_bodies, 3). Defaults to None.
            body_ids: Body ids. Defaults to None (all bodies).
            env_ids: Environment ids. Defaults to None (all environments).
            is_global: Whether the forces and torques are applied in the global frame. Defaults to False.
        """
        self.add_forces_and_torques_index(
            forces=forces,
            torques=torques,
            positions=positions,
            body_ids=body_ids,
            env_ids=env_ids,
            is_global=is_global,
        )

    def set_forces_and_torques(
        self,
        forces: wp.array | torch.Tensor | None = None,
        torques: wp.array | torch.Tensor | None = None,
        positions: wp.array | torch.Tensor | None = None,
        body_ids: wp.array | torch.Tensor | None = None,
        env_ids: wp.array | torch.Tensor | None = None,
        is_global: bool = False,
    ) -> None:
        """Set forces and torques (deprecated, use set_forces_and_torques_index).

        Args:
            forces: Forces. (num_envs, num_bodies, 3). Defaults to None.
            torques: Torques. (num_envs, num_bodies, 3). Defaults to None.
            positions: Positions. (num_envs, num_bodies, 3). Defaults to None.
            body_ids: Body ids. Defaults to None (all bodies).
            env_ids: Environment ids. Defaults to None (all environments).
            is_global: Whether the forces and torques are applied in the global frame. Defaults to False.
        """
        self.set_forces_and_torques_index(
            forces=forces,
            torques=torques,
            positions=positions,
            body_ids=body_ids,
            env_ids=env_ids,
            is_global=is_global,
        )

    # -- Index/Mask method variants --

    def add_forces_and_torques_index(
        self,
        forces: wp.array | torch.Tensor | None = None,
        torques: wp.array | torch.Tensor | None = None,
        positions: wp.array | torch.Tensor | None = None,
        body_ids: torch.Tensor | None = None,
        env_ids: torch.Tensor | None = None,
        is_global: bool = False,
    ) -> None:
        """Add forces and torques by index (mock - sets active/dirty flags)."""
        if forces is not None or torques is not None:
            self._active = True
            self._dirty = True

    def add_forces_and_torques_mask(
        self,
        forces: wp.array | torch.Tensor | None = None,
        torques: wp.array | torch.Tensor | None = None,
        positions: wp.array | torch.Tensor | None = None,
        body_mask: wp.array | torch.Tensor | None = None,
        env_mask: wp.array | torch.Tensor | None = None,
        is_global: bool = False,
    ) -> None:
        """Add forces and torques by mask (mock - sets active/dirty flags)."""
        if forces is not None or torques is not None:
            self._active = True
            self._dirty = True

    def set_forces_and_torques_index(
        self,
        forces: wp.array | torch.Tensor | None = None,
        torques: wp.array | torch.Tensor | None = None,
        positions: wp.array | torch.Tensor | None = None,
        body_ids: wp.array | torch.Tensor | None = None,
        env_ids: wp.array | torch.Tensor | None = None,
        is_global: bool = False,
    ) -> None:
        """Set forces and torques by index (mock - sets active/dirty flags)."""
        if forces is not None or torques is not None:
            self._active = True
            self._dirty = True

    def set_forces_and_torques_mask(
        self,
        forces: wp.array | torch.Tensor | None = None,
        torques: wp.array | torch.Tensor | None = None,
        positions: wp.array | torch.Tensor | None = None,
        body_mask: wp.array | torch.Tensor | None = None,
        env_mask: wp.array | torch.Tensor | None = None,
        is_global: bool = False,
    ) -> None:
        """Set forces and torques by mask (mock - sets active/dirty flags)."""
        if forces is not None or torques is not None:
            self._active = True
            self._dirty = True

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self, env_ids: wp.array | torch.Tensor | None = None, env_mask: wp.array | None = None) -> None:
        """Reset all 7 buffers (5 input + 2 output) and clear all flags.

        Args:
            env_ids: Environment ids to reset. Defaults to None (all environments).
            env_mask: Environment mask to reset. Defaults to None (all environments).
        """
        if env_ids is None and env_mask is None:
            # Full reset: zero all 7 buffers and clear flags
            self._global_force_w.zero_()
            self._global_torque_w.zero_()
            self._global_force_at_com_w.zero_()
            self._local_force_b.zero_()
            self._local_torque_b.zero_()
            self._out_force_b.zero_()
            self._out_torque_b.zero_()
            self._active = False
            self._dirty = False
        else:
            # For partial reset, just zero the specified environments across all 7 buffers
            if isinstance(env_ids, torch.Tensor):
                indices = wp.from_torch(env_ids.to(torch.int32), dtype=wp.int32)
            elif isinstance(env_ids, list):
                indices = wp.array(env_ids, dtype=wp.int32, device=self.device)
            else:
                indices = env_ids

            # Zero all 7 buffers for the specified environments
            # Use torch views for the indexing operation
            for buf in [
                self._global_force_w,
                self._global_torque_w,
                self._global_force_at_com_w,
                self._local_force_b,
                self._local_torque_b,
                self._out_force_b,
                self._out_torque_b,
            ]:
                buf_torch = wp.to_torch(buf)
                if isinstance(env_ids, torch.Tensor):
                    buf_torch[env_ids.long()] = 0.0
                else:
                    idx_torch = wp.to_torch(indices).long()
                    buf_torch[idx_torch] = 0.0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_composed(self):
        """Compose input buffers into output buffers if dirty."""
        if self._dirty:
            self.compose_to_body_frame()
