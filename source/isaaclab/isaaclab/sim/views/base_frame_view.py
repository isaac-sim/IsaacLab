# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Abstract base class for batched prim transform views."""

from __future__ import annotations

import abc
import warnings

import warp as wp

from isaaclab.utils.warp import ProxyArray


class BaseFrameView(abc.ABC):
    """Abstract interface for reading and writing world-space transforms of multiple prims.

    Backend-specific implementations (USD/Fabric, Newton GPU state, etc.) subclass
    this to provide efficient batched pose queries.  The factory
    :class:`~isaaclab.sim.views.FrameView` selects the correct
    implementation at runtime based on the active physics backend.

    All getters return :class:`~isaaclab.utils.warp.ProxyArray`.  Setters accept ``wp.array``.
    """

    @property
    @abc.abstractmethod
    def count(self) -> int:
        """Number of prims in this view."""
        ...

    @property
    @abc.abstractmethod
    def device(self) -> str:
        """Device where arrays are allocated (``"cpu"`` or ``"cuda:0"``)."""
        ...

    @abc.abstractmethod
    def get_world_poses(self, indices: wp.array | None = None) -> tuple[ProxyArray, ProxyArray]:
        """Get world-space positions and orientations for prims in the view.

        Args:
            indices: Subset of prims to query.  ``None`` means all prims.

        Returns:
            A tuple ``(positions, orientations)`` of :class:`~isaaclab.utils.warp.ProxyArray`
            wrappers. Use ``.warp`` for the underlying ``wp.array`` or ``.torch`` for a
            cached zero-copy ``torch.Tensor`` view.
        """
        ...

    @abc.abstractmethod
    def set_world_poses(
        self,
        positions: wp.array | None = None,
        orientations: wp.array | None = None,
        indices: wp.array | None = None,
    ) -> None:
        """Set world-space positions and/or orientations for prims in the view.

        Args:
            positions: World-space positions ``(M, 3)``. ``None`` leaves positions unchanged.
            orientations: World-space quaternions ``(M, 4)``. ``None`` leaves orientations unchanged.
            indices: Subset of prims to update.  ``None`` means all prims.
        """
        ...

    @abc.abstractmethod
    def get_local_poses(self, indices: wp.array | None = None) -> tuple[ProxyArray, ProxyArray]:
        """Get local-space positions and orientations for prims in the view.

        Args:
            indices: Subset of prims to query.  ``None`` means all prims.

        Returns:
            A tuple ``(translations, orientations)`` of :class:`~isaaclab.utils.warp.ProxyArray`
            wrappers. Use ``.warp`` for the underlying ``wp.array`` or ``.torch`` for a
            cached zero-copy ``torch.Tensor`` view.
        """
        ...

    @abc.abstractmethod
    def set_local_poses(
        self,
        translations: wp.array | None = None,
        orientations: wp.array | None = None,
        indices: wp.array | None = None,
    ) -> None:
        """Set local-space translations and/or orientations for prims in the view.

        Args:
            translations: Local-space translations ``(M, 3)``. ``None`` leaves translations unchanged.
            orientations: Local-space quaternions ``(M, 4)``. ``None`` leaves orientations unchanged.
            indices: Subset of prims to update.  ``None`` means all prims.
        """
        ...

    @abc.abstractmethod
    def get_local_scales(self, indices: wp.array | None = None) -> wp.array:
        """Get local-space scales for prims in the view.

        Returns the per-prim scale as stored in local space (``xformOp:scale``
        for USD, decomposition of ``localMatrix`` for Fabric).

        Args:
            indices: Subset of prims to query.  ``None`` means all prims.

        Returns:
            A :class:`~isaaclab.utils.warp.ProxyArray` of shape ``(M, 3)``.
        """
        ...

    @abc.abstractmethod
    def set_local_scales(self, scales: wp.array, indices: wp.array | None = None) -> None:
        """Set local-space scales for prims in the view.

        Writes the per-prim scale in local space.  For the Fabric backend this
        marks the world matrix as dirty (will be re-propagated on next read).

        Args:
            scales: Scales ``(M, 3)`` as ``wp.array``.
            indices: Subset of prims to update.  ``None`` means all prims.
        """
        ...

    @abc.abstractmethod
    def get_world_scales(self, indices: wp.array | None = None) -> wp.array:
        """Get world-space (composed) scales for prims in the view.

        Returns the effective scale in world space (``parent_scale * local_scale``).

        Args:
            indices: Subset of prims to query.  ``None`` means all prims.

        Returns:
            A ``wp.array`` of shape ``(M, 3)``.
        """
        ...

    @abc.abstractmethod
    def set_world_scales(self, scales: wp.array, indices: wp.array | None = None) -> None:
        """Set world-space (composed) scales for prims in the view.

        Writes the effective scale in world space.  For the Fabric backend this
        marks the local matrix as dirty (will be re-propagated on next read).

        Args:
            scales: Scales ``(M, 3)`` as ``wp.array``.
            indices: Subset of prims to update.  ``None`` means all prims.
        """
        ...

    # ------------------------------------------------------------------
    # Deprecated -- use get/set_local_scales or get/set_world_scales
    # ------------------------------------------------------------------

    def get_scales(self, indices: wp.array | None = None) -> wp.array:
        """Get scales for prims in the view.

        .. deprecated::
            Use :meth:`get_local_scales` or :meth:`get_world_scales` instead.
            This method calls ``get_local_scales`` for USD backends and
            ``get_world_scales`` for Fabric backends (preserving legacy behavior).

        Args:
            indices: Subset of prims to query.  ``None`` means all prims.

        Returns:
            A ``wp.array`` of shape ``(M, 3)``.
        """
        warnings.warn(
            "get_scales() is deprecated. Use get_local_scales() or get_world_scales() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._get_scales_default(indices)

    def set_scales(self, scales: wp.array, indices: wp.array | None = None) -> None:
        """Set scales for prims in the view.

        .. deprecated::
            Use :meth:`set_local_scales` or :meth:`set_world_scales` instead.
            This method calls ``set_local_scales`` for USD backends and
            ``set_world_scales`` for Fabric backends (preserving legacy behavior).

        Args:
            scales: Scales ``(M, 3)`` as ``wp.array``.
            indices: Subset of prims to update.  ``None`` means all prims.
        """
        warnings.warn(
            "set_scales() is deprecated. Use set_local_scales() or set_world_scales() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._set_scales_default(scales, indices)

    @abc.abstractmethod
    def _get_scales_default(self, indices: wp.array | None = None) -> wp.array:
        """Backend-specific default for deprecated get_scales()."""
        ...

    @abc.abstractmethod
    def _set_scales_default(self, scales: wp.array, indices: wp.array | None = None) -> None:
        """Backend-specific default for deprecated set_scales()."""
        ...
