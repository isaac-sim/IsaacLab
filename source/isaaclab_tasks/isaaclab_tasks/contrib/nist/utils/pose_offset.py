# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Cached rigid-body pose offset.

A small configclass that pre-caches a translation+quaternion offset (and its
inverse) as device tensors and exposes ``apply`` / ``combine`` / ``subtract``
helpers for composing the offset against an asset's root frame or arbitrary
parent frames. Pure rigid-body math; no domain semantics. Used by both
manipulation (assembly keypoints, gripper grasp offsets) and any future
locomotion/terrain code that needs to author static pose transforms.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import warp as wp

import isaaclab.utils.math as math_utils
from isaaclab.utils.configclass import configclass

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject


@configclass
class Offset:
    """A position + quaternion offset relative to an asset root frame.

    Args:
        pos: Translation offset [m].
        quat: Orientation offset in (x, y, z, w).
    """

    pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    quat: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)

    @property
    def pose(self) -> tuple[float, float, float, float, float, float, float]:
        return self.pos + self.quat

    def __post_init__(self):
        self._pos_t: torch.Tensor | None = None
        self._quat_t: torch.Tensor | None = None
        self._inv_pos_t: torch.Tensor | None = None
        self._inv_quat_t: torch.Tensor | None = None

    def _ensure_tensors(self, device: str | torch.device):
        """Materialize and cache forward + inverse tensors on first use."""
        if self._pos_t is None:
            self._pos_t = torch.tensor([self.pos], device=device)
            self._quat_t = torch.tensor([self.quat], device=device)
            self._inv_quat_t = math_utils.quat_inv(self._quat_t)
            self._inv_pos_t = -math_utils.quat_apply(self._inv_quat_t, self._pos_t)

    def pos_t(self, device: str | torch.device) -> torch.Tensor:
        """Cached position tensor, shape (1, 3)."""
        self._ensure_tensors(device)
        assert self._pos_t is not None
        return self._pos_t

    def quat_t(self, device: str | torch.device) -> torch.Tensor:
        """Cached quaternion tensor, shape (1, 4)."""
        self._ensure_tensors(device)
        assert self._quat_t is not None
        return self._quat_t

    def apply(self, root: RigidObject | Articulation) -> tuple[torch.Tensor, torch.Tensor]:
        """Transform this offset into world frame using the asset's current root pose.

        Args:
            root: The asset whose root pose defines the parent frame.

        Returns:
            A tuple of (position, quaternion) tensors in world frame,
            each with shape (num_envs, 3) and (num_envs, 4) respectively.
        """
        root_pos = wp.to_torch(root.data.root_pos_w)
        root_quat = wp.to_torch(root.data.root_quat_w)
        self._ensure_tensors(root_pos.device)
        return math_utils.combine_frame_transforms(
            root_pos, root_quat, self._pos_t.expand(root_pos.shape[0], -1), self._quat_t.expand(root_pos.shape[0], -1)
        )

    def combine(self, pos: torch.Tensor, quat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compose this offset on top of an arbitrary parent frame.

        Unlike :meth:`apply`, the parent frame is given directly as tensors
        rather than read from an asset.

        Args:
            pos: Parent frame positions [m], shape (N, 3).
            quat: Parent frame quaternions (x, y, z, w), shape (N, 4).

        Returns:
            A tuple of (position, quaternion) tensors for the composed frame.
        """
        self._ensure_tensors(pos.device)
        return math_utils.combine_frame_transforms(
            pos, quat, self._pos_t.expand(pos.shape[0], -1), self._quat_t.expand(pos.shape[0], -1)
        )

    def subtract(self, pos: torch.Tensor, quat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Right-multiply the given frame by this offset's inverse.

        Solves for the root frame A such that ``A.combine(self) == (pos, quat)``.
        Computes ``T_target * T_self^{-1}``.

        Args:
            pos: Target frame positions [m], shape (N, 3).
            quat: Target frame quaternions (x, y, z, w), shape (N, 4).

        Returns:
            A tuple of (position, quaternion) tensors for the solved root frame.
        """
        self._ensure_tensors(pos.device)
        return math_utils.combine_frame_transforms(
            pos, quat, self._inv_pos_t.expand(pos.shape[0], -1), self._inv_quat_t.expand(pos.shape[0], -1)
        )
