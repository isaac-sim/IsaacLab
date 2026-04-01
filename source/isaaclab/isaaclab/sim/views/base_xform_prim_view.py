# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Abstract base class for batched prim transform views."""

from __future__ import annotations

import abc
from collections.abc import Sequence

import torch


class BaseXformPrimView(abc.ABC):
    """Abstract interface for reading and writing world-space transforms of multiple prims.

    Backend-specific implementations (USD/Fabric, Newton GPU state, etc.) subclass
    this to provide efficient batched pose queries.  The factory
    :class:`~isaaclab.sim.views.XformPrimViewFactory` selects the correct
    implementation at runtime based on the active physics backend.
    """

    @property
    @abc.abstractmethod
    def count(self) -> int:
        """Number of prims in this view."""
        ...

    @abc.abstractmethod
    def get_world_poses(
        self, indices: Sequence[int] | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Get world-space positions and orientations for prims in the view.

        Args:
            indices: Subset of prims to query.  ``None`` means all prims.

        Returns:
            A tuple ``(positions (M, 3), orientations (M, 4))``.
        """
        ...

    @abc.abstractmethod
    def set_world_poses(
        self,
        positions: torch.Tensor | None = None,
        orientations: torch.Tensor | None = None,
        indices: Sequence[int] | None = None,
    ) -> None:
        """Set world-space positions and/or orientations for prims in the view.

        Args:
            positions: World-space positions ``(M, 3)``. ``None`` leaves positions unchanged.
            orientations: World-space quaternions ``(M, 4)``. ``None`` leaves orientations unchanged.
            indices: Subset of prims to update.  ``None`` means all prims.
        """
        ...

    @abc.abstractmethod
    def get_local_poses(
        self, indices: Sequence[int] | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Get local-space positions and orientations for prims in the view.

        Args:
            indices: Subset of prims to query.  ``None`` means all prims.

        Returns:
            A tuple ``(translations (M, 3), orientations (M, 4))``.
        """
        ...

    @abc.abstractmethod
    def set_local_poses(
        self,
        translations: torch.Tensor | None = None,
        orientations: torch.Tensor | None = None,
        indices: Sequence[int] | None = None,
    ) -> None:
        """Set local-space translations and/or orientations for prims in the view.

        Args:
            translations: Local-space translations ``(M, 3)``. ``None`` leaves translations unchanged.
            orientations: Local-space quaternions ``(M, 4)``. ``None`` leaves orientations unchanged.
            indices: Subset of prims to update.  ``None`` means all prims.
        """
        ...

    @abc.abstractmethod
    def get_scales(self, indices: Sequence[int] | None = None) -> torch.Tensor:
        """Get scales for prims in the view.

        Args:
            indices: Subset of prims to query.  ``None`` means all prims.

        Returns:
            A tensor of shape ``(M, 3)``.
        """
        ...

    @abc.abstractmethod
    def set_scales(self, scales: torch.Tensor, indices: Sequence[int] | None = None) -> None:
        """Set scales for prims in the view.

        Args:
            scales: Scales ``(M, 3)``.
            indices: Subset of prims to update.  ``None`` means all prims.
        """
        ...
