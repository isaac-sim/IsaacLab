# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import torch


class XformBackend(Protocol):
    """Protocol defining the interface for :class:`XformPrimView` transform backends.

    Implementations provide read/write access to prim transforms through either
    the USD or Fabric data path.  :class:`XformPrimView` delegates all transform
    operations to a *primary* backend and optionally replicates writes to one or
    more *sync* backends.
    """

    def initialize(self) -> None:
        """Perform any deferred initialisation required by the backend."""
        ...

    def set_world_poses(
        self,
        positions: torch.Tensor | None = None,
        orientations: torch.Tensor | None = None,
        indices: Sequence[int] | None = None,
    ) -> None:
        """Set world-space poses for the managed prims.

        Args:
            positions: World-space positions, shape ``(M, 3)`` [m].
            orientations: World-space quaternions ``(x, y, z, w)``, shape ``(M, 4)``.
            indices: Subset of prim indices to update.  ``None`` means all.
        """
        ...

    def get_world_poses(
        self,
        indices: Sequence[int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return world-space ``(positions, orientations)`` for the managed prims.

        Args:
            indices: Subset of prim indices to query.  ``None`` means all.

        Returns:
            ``(positions, orientations)`` with shapes ``(M, 3)`` and ``(M, 4)``.
        """
        ...

    def set_local_poses(
        self,
        translations: torch.Tensor | None = None,
        orientations: torch.Tensor | None = None,
        indices: Sequence[int] | None = None,
    ) -> None:
        """Set local-space poses (relative to each prim's parent).

        Args:
            translations: Local-space translations, shape ``(M, 3)`` [m].
            orientations: Local-space quaternions ``(x, y, z, w)``, shape ``(M, 4)``.
            indices: Subset of prim indices to update.  ``None`` means all.
        """
        ...

    def get_local_poses(
        self,
        indices: Sequence[int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return local-space ``(translations, orientations)`` for the managed prims.

        Args:
            indices: Subset of prim indices to query.  ``None`` means all.

        Returns:
            ``(translations, orientations)`` with shapes ``(M, 3)`` and ``(M, 4)``.
        """
        ...

    def set_scales(
        self,
        scales: torch.Tensor,
        indices: Sequence[int] | None = None,
    ) -> None:
        """Set scales for the managed prims.

        Args:
            scales: Scales, shape ``(M, 3)``.
            indices: Subset of prim indices to update.  ``None`` means all.
        """
        ...

    def get_scales(
        self,
        indices: Sequence[int] | None = None,
    ) -> torch.Tensor:
        """Return scales for the managed prims.

        Args:
            indices: Subset of prim indices to query.  ``None`` means all.

        Returns:
            Scales tensor of shape ``(M, 3)``.
        """
        ...
