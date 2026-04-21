# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import torch
import warp as wp

from isaaclab.utils.warp.kernels import (
    add_forces_to_dual_buffers_index,
    add_forces_to_dual_buffers_mask,
    add_raw_wrench_buffers,
    compose_wrench_to_body_frame,
    reset_wrench_composer_index,
    reset_wrench_composer_mask,
    set_forces_to_dual_buffers_index,
    set_forces_to_dual_buffers_mask,
)

if TYPE_CHECKING:
    from isaaclab.assets import BaseArticulation, BaseRigidObject, BaseRigidObjectCollection


class WrenchComposer:
    def __init__(self, asset: BaseArticulation | BaseRigidObject | BaseRigidObjectCollection) -> None:
        """Wrench composer with dual-buffer architecture.

        This class composes forces and torques applied to rigid bodies. Forces and torques can be
        specified in either the global (world) frame or the local (body) frame. Internally, they are
        stored in separate global and local input buffers. When the final composed wrench is needed,
        the global contributions are rotated into the body frame and combined with the local
        contributions to produce the output force and torque expressed in the body frame.

        The dual-buffer architecture uses five input buffers:

        - ``global_force_w``: Global forces [N] (world frame).
        - ``global_torque_w``: Global torques [N·m] (world frame), including moment contributions
          from positional forces (``cross(P, F)``).
        - ``global_force_at_com_w``: Global forces [N] applied at the body's CoM (world frame, no positional torque).
        - ``local_force_b``: Local forces [N] (body frame).
        - ``local_torque_b``: Local torques [N·m] (body frame).

        And two output buffers:

        - ``out_force_b``: Composed force [N] in body frame.
        - ``out_torque_b``: Composed torque [N·m] in body frame.

        Args:
            asset: Asset to use.
        """
        self.num_envs = asset.num_instances
        # Avoid isinstance to prevent circular import issues; check by attribute presence instead.
        if hasattr(asset, "num_bodies"):
            self.num_bodies = asset.num_bodies
        else:
            raise ValueError(f"Unsupported asset type: {asset.__class__.__name__}")
        self.device = asset.device
        self._asset = asset
        self._active = False
        self._dirty = False
        if hasattr(self._asset.data, "body_com_pos_w"):
            self._get_com_pos_fn = lambda a=self._asset: a.data.body_com_pos_w
        else:
            raise ValueError(f"Unsupported asset type: {self._asset.__class__.__name__}")
        if hasattr(self._asset.data, "body_link_quat_w"):
            self._get_link_quat_fn = lambda a=self._asset: a.data.body_link_quat_w
        else:
            raise ValueError(f"Unsupported asset type: {self._asset.__class__.__name__}")

        # -- Input buffers (5 total) --
        self._global_force_w = wp.zeros((self.num_envs, self.num_bodies), dtype=wp.vec3f, device=self.device)
        self._global_torque_w = wp.zeros((self.num_envs, self.num_bodies), dtype=wp.vec3f, device=self.device)
        self._global_force_at_com_w = wp.zeros((self.num_envs, self.num_bodies), dtype=wp.vec3f, device=self.device)
        self._local_force_b = wp.zeros((self.num_envs, self.num_bodies), dtype=wp.vec3f, device=self.device)
        self._local_torque_b = wp.zeros((self.num_envs, self.num_bodies), dtype=wp.vec3f, device=self.device)

        # -- Output buffers (2 total) --
        self._out_force_b = wp.zeros((self.num_envs, self.num_bodies), dtype=wp.vec3f, device=self.device)
        self._out_torque_b = wp.zeros((self.num_envs, self.num_bodies), dtype=wp.vec3f, device=self.device)

        # -- Index / mask helper arrays --
        self._ALL_ENV_INDICES = wp.array(np.arange(self.num_envs, dtype=np.int32), dtype=wp.int32, device=self.device)
        self._ALL_BODY_INDICES = wp.array(
            np.arange(self.num_bodies, dtype=np.int32), dtype=wp.int32, device=self.device
        )
        self._ALL_ENV_MASK = wp.ones((self.num_envs), dtype=wp.bool, device=self.device)
        self._ALL_BODY_MASK = wp.ones((self.num_bodies), dtype=wp.bool, device=self.device)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def active(self) -> bool:
        """Whether the wrench composer is active (has pending forces/torques).

        Set to ``True`` when any ``add_*`` or ``set_*`` method writes data. Cleared only by a
        full :meth:`reset` call (no arguments). Partial resets (with ``env_ids`` or ``env_mask``)
        do **not** clear this flag because checking whether all environments are zero would
        require scanning the buffers, defeating the purpose of a cheap guard.

        This means the flag may remain ``True`` even if all buffers are zero after partial resets.
        This is by design: the cost of an unnecessary compose + apply on zero data is negligible
        compared to scanning the buffers every frame.
        """
        return self._active

    @property
    def global_force_w(self) -> wp.array:
        """Global force buffer [N] (world frame), dtype ``wp.vec3f``. Shape: ``(num_envs, num_bodies)``.

        .. note::
            This returns the underlying buffer reference for read-only inspection. Writing to it
            directly bypasses the dirty flag and may produce stale output buffers. Use the
            ``add_*`` or ``set_*`` methods to modify forces.
        """
        return self._global_force_w

    @property
    def global_torque_w(self) -> wp.array:
        """Global torque buffer [N·m] (world frame), dtype ``wp.vec3f``. Shape: ``(num_envs, num_bodies)``.

        Stores user-supplied torques plus moment contributions from positional forces (``cross(P, F)``).

        .. note::
            Read-only reference. See :attr:`global_force_w` for caveats on direct writes.
        """
        return self._global_torque_w

    @property
    def global_force_at_com_w(self) -> wp.array:
        """Global force at body's CoM buffer [N] (world frame, no positional torque).

        dtype ``wp.vec3f``. Shape: ``(num_envs, num_bodies)``.

        .. note::
            Read-only reference. See :attr:`global_force_w` for caveats on direct writes.
        """
        return self._global_force_at_com_w

    @property
    def local_force_b(self) -> wp.array:
        """Local force buffer [N] (body frame), dtype ``wp.vec3f``. Shape: ``(num_envs, num_bodies)``.

        .. note::
            Read-only reference. See :attr:`global_force_w` for caveats on direct writes.
        """
        return self._local_force_b

    @property
    def local_torque_b(self) -> wp.array:
        """Local torque buffer [N·m] (body frame), dtype ``wp.vec3f``. Shape: ``(num_envs, num_bodies)``.

        .. note::
            Read-only reference. See :attr:`global_force_w` for caveats on direct writes.
        """
        return self._local_torque_b

    @property
    def out_force_b(self) -> wp.array:
        """Composed output force [N] in the body frame, dtype ``wp.vec3f``. Shape: ``(num_envs, num_bodies)``.

        Triggers composition from input buffers if dirty.
        """
        self._ensure_composed()
        return self._out_force_b

    @property
    def out_torque_b(self) -> wp.array:
        """Composed output torque [N·m] in the body frame, dtype ``wp.vec3f``. Shape: ``(num_envs, num_bodies)``.

        Triggers composition from input buffers if dirty.
        """
        self._ensure_composed()
        return self._out_torque_b

    @property
    def composed_force(self) -> wp.array:
        """Composed force at the body frame, dtype ``wp.vec3f``. Shape: ``(num_envs, num_bodies)``.

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
        """Composed torque at the body frame, dtype ``wp.vec3f``. Shape: ``(num_envs, num_bodies)``.

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
    # Public methods
    # ------------------------------------------------------------------

    def add_forces_and_torques_index(
        self,
        forces: wp.array | torch.Tensor | None = None,
        torques: wp.array | torch.Tensor | None = None,
        positions: wp.array | torch.Tensor | None = None,
        body_ids: torch.Tensor | None = None,
        env_ids: torch.Tensor | None = None,
        is_global: bool = False,
    ):
        """Add forces and torques into the input buffers using index-based selection.

        Accumulates onto whatever is already in the buffers. The result is always composed into the
        body frame when the output properties are accessed.

        Args:
            forces: Forces [N]. Shape: (len(env_ids), len(body_ids), 3). Defaults to None.
            torques: Torques [N·m]. Shape: (len(env_ids), len(body_ids), 3). Defaults to None.
            positions: The positions [m] at which forces act. If `is_global` is True, these are global
                positions expressed in the world frame. If `is_global` is False, these are offsets from the
                body's CoM expressed in the body frame. If None, forces are assumed to act at the body's
                CoM, independent of the `is_global` flag.
                Shape: (len(env_ids), len(body_ids), 3). Defaults to None.
            body_ids: Body indices. Defaults to None (all bodies).
            env_ids: Environment indices. Defaults to None (all environments).
            is_global: Whether the forces and torques are expressed in the global world frame or the local body frame.
                Defaults to False.
        """
        env_ids = self._resolve_env_ids(env_ids)
        body_ids = self._resolve_body_ids(body_ids)
        if forces is None and torques is None:
            warnings.warn(
                "No forces or torques provided. No force will be added.",
                UserWarning,
                stacklevel=2,
            )
            return

        self._active = True
        self._dirty = True

        wp.launch(
            add_forces_to_dual_buffers_index,
            dim=(env_ids.shape[0], body_ids.shape[0]),
            inputs=[
                env_ids,
                body_ids,
                forces,
                torques,
                positions,
                self._global_force_w,
                self._global_torque_w,
                self._global_force_at_com_w,
                self._local_force_b,
                self._local_torque_b,
                is_global,
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
        """Set forces and torques into the input buffers using index-based selection.

        Resets the specified environments first, then writes the new values. This replaces any
        previously accumulated forces/torques for the targeted environments while leaving other
        environments untouched.

        Args:
            forces: Forces [N]. Shape: (len(env_ids), len(body_ids), 3). Defaults to None.
            torques: Torques [N·m]. Shape: (len(env_ids), len(body_ids), 3). Defaults to None.
            positions: The positions [m] at which forces act. If `is_global` is True, these are global
                positions expressed in the world frame. If `is_global` is False, these are offsets from the
                body's CoM expressed in the body frame. If None, forces are assumed to act at the body's
                CoM, independent of the `is_global` flag.
                Shape: (len(env_ids), len(body_ids), 3). Defaults to None.
            body_ids: Body indices. Defaults to None (all bodies).
            env_ids: Environment indices. Defaults to None (all environments).
            is_global: Whether the forces and torques are expressed in the global world frame or the local body frame.
                Defaults to False.
        """
        env_ids = self._resolve_env_ids(env_ids)
        body_ids = self._resolve_body_ids(body_ids)
        if forces is None and torques is None:
            warnings.warn(
                "No forces or torques provided. No force will be set.",
                UserWarning,
                stacklevel=2,
            )
            return

        # Clear input buffers for the targeted environments before writing
        self.reset(env_ids=env_ids)

        self._active = True
        self._dirty = True

        wp.launch(
            set_forces_to_dual_buffers_index,
            dim=(env_ids.shape[0], body_ids.shape[0]),
            inputs=[
                env_ids,
                body_ids,
                forces,
                torques,
                positions,
                self._global_force_w,
                self._global_torque_w,
                self._global_force_at_com_w,
                self._local_force_b,
                self._local_torque_b,
                is_global,
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
        """Add forces and torques into the input buffers using mask-based selection.

        Accumulates onto whatever is already in the buffers.

        Args:
            forces: Forces [N]. Shape: (num_envs, num_bodies, 3). Defaults to None.
            torques: Torques [N·m]. Shape: (num_envs, num_bodies, 3). Defaults to None.
            positions: The positions [m] at which forces act. If `is_global` is True, these are global
                positions expressed in the world frame. If `is_global` is False, these are offsets from the
                body's CoM expressed in the body frame. If None, forces are assumed to act at the body's
                CoM, independent of the `is_global` flag.
                Shape: (num_envs, num_bodies, 3). Defaults to None.
            body_mask: Body mask. Shape: (num_bodies,). Defaults to None (all bodies).
            env_mask: Environment mask. Shape: (num_envs,). Defaults to None (all environments).
            is_global: Whether the forces and torques are expressed in the global world frame or the local body frame.
                Defaults to False.
        """
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

        self._active = True
        self._dirty = True

        wp.launch(
            add_forces_to_dual_buffers_mask,
            dim=(self.num_envs, self.num_bodies),
            inputs=[
                env_mask,
                body_mask,
                forces,
                torques,
                positions,
                self._global_force_w,
                self._global_torque_w,
                self._global_force_at_com_w,
                self._local_force_b,
                self._local_torque_b,
                is_global,
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
        """Set forces and torques into the input buffers using mask-based selection.

        Resets the masked environments first, then writes the new values. This replaces any
        previously accumulated forces/torques for the masked environments while leaving other
        environments untouched.

        Args:
            forces: Forces [N]. Shape: (num_envs, num_bodies, 3). Defaults to None.
            torques: Torques [N·m]. Shape: (num_envs, num_bodies, 3). Defaults to None.
            positions: The positions [m] at which forces act. If `is_global` is True, these are global
                positions expressed in the world frame. If `is_global` is False, these are offsets from the
                body's CoM expressed in the body frame. If None, forces are assumed to act at the body's
                CoM, independent of the `is_global` flag.
                Shape: (num_envs, num_bodies, 3). Defaults to None.
            body_mask: Body mask. Shape: (num_bodies,). Defaults to None (all bodies).
            env_mask: Environment mask. Shape: (num_envs,). Defaults to None (all environments).
            is_global: Whether the forces and torques are expressed in the global world frame or the local body frame.
                Defaults to False.
        """
        if env_mask is None:
            env_mask = self._ALL_ENV_MASK
        if body_mask is None:
            body_mask = self._ALL_BODY_MASK
        if forces is None and torques is None:
            warnings.warn(
                "No forces or torques provided. No force will be set.",
                UserWarning,
                stacklevel=2,
            )
            return

        # Clear input buffers for the masked environments before writing
        self.reset(env_mask=env_mask)

        self._active = True
        self._dirty = True

        wp.launch(
            set_forces_to_dual_buffers_mask,
            dim=(self.num_envs, self.num_bodies),
            inputs=[
                env_mask,
                body_mask,
                forces,
                torques,
                positions,
                self._global_force_w,
                self._global_torque_w,
                self._global_force_at_com_w,
                self._local_force_b,
                self._local_torque_b,
                is_global,
            ],
            device=self.device,
        )

    def add_raw_buffers_from(self, other: WrenchComposer):
        """Add another composer's raw input buffers into this composer's input buffers.

        This performs element-wise addition of all five input buffers from ``other`` into ``self``.
        Useful for combining wrenches from multiple sources before composition.

        Args:
            other: Another WrenchComposer whose input buffers will be added into this one.
        """
        if not other._active:
            return
        if __debug__:
            if other.num_envs != self.num_envs or other.num_bodies != self.num_bodies:
                raise ValueError(
                    f"Cannot add buffers from composer with shape ({other.num_envs}, {other.num_bodies}) "
                    f"into composer with shape ({self.num_envs}, {self.num_bodies})."
                )

        self._active = True
        self._dirty = True

        wp.launch(
            add_raw_wrench_buffers,
            dim=(self.num_envs, self.num_bodies),
            inputs=[
                other._global_force_w,
                other._global_torque_w,
                other._global_force_at_com_w,
                other._local_force_b,
                other._local_torque_b,
                self._global_force_w,
                self._global_torque_w,
                self._global_force_at_com_w,
                self._local_force_b,
                self._local_torque_b,
            ],
            device=self.device,
        )

    def compose_to_body_frame(self):
        """Compose the five input buffers into the two output buffers in body frame.

        This corrects world-frame torques for the body's CoM position, rotates global forces and torques into the
        body frame, then adds local-frame contributions. After this call, ``out_force_b`` and ``out_torque_b``
        contain the final composed wrench.

        The dirty flag is cleared after composition.
        """
        com_pos_w = self._get_com_pos_fn()
        link_quat_w = self._get_link_quat_fn()

        wp.launch(
            compose_wrench_to_body_frame,
            dim=(self.num_envs, self.num_bodies),
            inputs=[
                self._global_force_w,
                self._global_torque_w,
                self._global_force_at_com_w,
                self._local_force_b,
                self._local_torque_b,
                com_pos_w,
                link_quat_w,
                self._out_force_b,
                self._out_torque_b,
            ],
            device=self.device,
        )
        self._dirty = False

    def reset(
        self,
        env_ids: wp.array | torch.Tensor | Sequence[int] | slice | None = None,
        env_mask: wp.array | None = None,
    ):
        """Reset the wrench composer buffers.

        With no arguments, zeros all seven buffers (5 input + 2 output) and clears all flags.
        With ``env_ids`` or ``env_mask``, performs a partial reset on the specified environments
        using the reset kernels.

        .. caution:: If both ``env_ids`` and ``env_mask`` are provided, ``env_mask`` takes precedence.

        Args:
            env_ids: Environment indices. Defaults to None (all environments).
            env_mask: Environment mask. Defaults to None (all environments).
        """
        if env_ids is None and env_mask is None:
            # Full reset: zero all 7 buffers
            self._global_force_w.zero_()
            self._global_torque_w.zero_()
            self._global_force_at_com_w.zero_()
            self._local_force_b.zero_()
            self._local_torque_b.zero_()
            self._out_force_b.zero_()
            self._out_torque_b.zero_()
            self._active = False
            self._dirty = False
        elif env_mask is not None:
            wp.launch(
                reset_wrench_composer_mask,
                dim=(self.num_envs, self.num_bodies),
                inputs=[
                    env_mask,
                    self._global_force_w,
                    self._global_torque_w,
                    self._global_force_at_com_w,
                    self._local_force_b,
                    self._local_torque_b,
                    self._out_force_b,
                    self._out_torque_b,
                ],
                device=self.device,
            )
            self._dirty = True
        else:
            # Partial reset via index
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
                    self._global_force_w,
                    self._global_torque_w,
                    self._global_force_at_com_w,
                    self._local_force_b,
                    self._local_torque_b,
                    self._out_force_b,
                    self._out_torque_b,
                ],
                device=self.device,
            )
            self._dirty = True

    # ------------------------------------------------------------------
    # Deprecated methods
    # ------------------------------------------------------------------

    def add_forces_and_torques(
        self,
        forces: wp.array | torch.Tensor | None = None,
        torques: wp.array | torch.Tensor | None = None,
        positions: wp.array | torch.Tensor | None = None,
        body_ids: torch.Tensor | None = None,
        env_ids: torch.Tensor | None = None,
        is_global: bool = False,
    ):
        """Deprecated, same as :meth:`add_forces_and_torques_index`.

        .. deprecated:: 4.5.33
            Use :meth:`add_forces_and_torques_index` instead.
        """
        warnings.warn(
            "The function 'add_forces_and_torques' is deprecated. Please use 'add_forces_and_torques_index' instead.",
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
        """Deprecated, same as :meth:`set_forces_and_torques_index`.

        .. deprecated:: 4.5.33
            Use :meth:`set_forces_and_torques_index` instead.
        """
        warnings.warn(
            "The function 'set_forces_and_torques' is deprecated. Please use 'set_forces_and_torques_index' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.set_forces_and_torques_index(forces, torques, positions, body_ids, env_ids, is_global)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_env_ids(self, env_ids: wp.array | torch.Tensor | list | slice | None) -> wp.array:
        """Resolve environment IDs to a warp int32 array.

        Args:
            env_ids: Environment indices as any supported type, or None for all environments.

        Returns:
            Warp array of int32 environment indices.

        Raises:
            TypeError: If ``env_ids`` is an unsupported type.
        """
        if env_ids is None:
            return self._ALL_ENV_INDICES
        # Check tensor types before slice comparison (tensor == slice crashes)
        if isinstance(env_ids, torch.Tensor):
            if env_ids.dtype == torch.int64:
                env_ids = env_ids.to(torch.int32)
            return wp.from_torch(env_ids.contiguous(), dtype=wp.int32)
        if isinstance(env_ids, wp.array):
            return env_ids
        if env_ids == slice(None):
            return self._ALL_ENV_INDICES
        if isinstance(env_ids, list):
            return wp.array(env_ids, dtype=wp.int32, device=self.device)
        raise TypeError(
            f"env_ids must be None, slice(None), list, torch.Tensor, or wp.array, got {type(env_ids).__name__}"
        )

    def _resolve_body_ids(self, body_ids: wp.array | torch.Tensor | list | slice | None) -> wp.array:
        """Resolve body IDs to a warp int32 array.

        Args:
            body_ids: Body indices as any supported type, or None for all bodies.

        Returns:
            Warp array of int32 body indices.

        Raises:
            TypeError: If ``body_ids`` is an unsupported type.
        """
        if body_ids is None:
            return self._ALL_BODY_INDICES
        if isinstance(body_ids, torch.Tensor):
            if body_ids.dtype == torch.int64:
                body_ids = body_ids.to(torch.int32)
            return wp.from_torch(body_ids.contiguous(), dtype=wp.int32)
        if isinstance(body_ids, wp.array):
            return body_ids
        if body_ids == slice(None):
            return self._ALL_BODY_INDICES
        if isinstance(body_ids, list):
            return wp.array(body_ids, dtype=wp.int32, device=self.device)
        raise TypeError(
            f"body_ids must be None, slice(None), list, torch.Tensor, or wp.array, got {type(body_ids).__name__}"
        )

    def _ensure_composed(self):
        """Compose input buffers into output buffers if dirty."""
        if self._dirty:
            self.compose_to_body_frame()
