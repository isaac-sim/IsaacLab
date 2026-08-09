# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Abstract base class for batched prim transform views."""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING

import warp as wp

from isaaclab.utils.warp import ProxyArray

if TYPE_CHECKING:
    from .xform_space_writer import FrameViewLocalSpaceWriter, FrameViewSpaceWriterBase, FrameViewWorldSpaceWriter


class BaseFrameView(abc.ABC):
    """Abstract interface for reading and writing transforms of multiple prims.

    Backend-specific implementations (USD/Fabric, Newton GPU state, etc.) subclass
    this to provide efficient batched pose queries.  The factory
    :class:`~isaaclab.sim.views.FrameView` selects the correct
    implementation at runtime based on the active physics backend.

    All getters return :class:`~isaaclab.utils.warp.ProxyArray`.  All writes go
    through the writer-scope API -- :meth:`xform_world_space_writer` or
    :meth:`xform_local_space_writer`:

    .. code-block:: python

        with view.xform_world_space_writer() as writer:
            writer.set_poses(positions=p, orientations=o)
            writer.set_scales(scales=s)
        # Derived-space matrices are recomputed and the writer scope is closed.

    Only one writer scope may be active per view at a time.  While a writer
    scope is active, the view-level getters
    (:meth:`get_world_poses`, :meth:`get_local_poses`,
    :meth:`get_world_scales`, :meth:`get_local_scales`) raise
    :class:`RuntimeError` -- use the writer's :meth:`~FrameViewSpaceWriterBase.get_poses`
    or :meth:`~FrameViewSpaceWriterBase.get_scales` inside the scope, or exit the
    scope first.
    """

    # Class-level default; instance-level value is set by the writer's
    # __enter__ / __exit__ to track the active scope on this view.
    _active_writer: FrameViewSpaceWriterBase | None = None

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

    def close(self) -> None:
        """Release backend state authored by this view. The view must not be used afterwards.

        The base implementation is a no-op; backends that author persistent
        state (e.g. the Fabric backend's per-view index attributes) override it.
        Backends also release best-effort when the view is garbage collected,
        but only an explicit :meth:`close` is deterministic -- collection
        timing is up to the interpreter.  Calling :meth:`close` more than once
        is safe.
        """

    # ------------------------------------------------------------------
    # Write scope -- recommended API for all transform writes.
    # ------------------------------------------------------------------

    def xform_world_space_writer(self) -> FrameViewWorldSpaceWriter:
        """Open a world-space write scope on this view (recommended write API).

        Inside the scope, :meth:`~FrameViewSpaceWriterBase.set_poses` /
        :meth:`~FrameViewSpaceWriterBase.set_scales` write world-space values.

        Returns:
            A :class:`~isaaclab.sim.views.FrameViewWorldSpaceWriter` context manager.

        Raises:
            RuntimeError: On ``__enter__``, if another writer is already active
                on this view.

        Example:
            .. code-block:: python

                with view.xform_world_space_writer() as w:
                    w.set_poses(positions=p, orientations=o)
                    w.set_scales(scales=s)
        """
        return self._make_world_space_writer()

    def xform_local_space_writer(self) -> FrameViewLocalSpaceWriter:
        """Open a local-space write scope on this view (recommended write API).

        Inside the scope, :meth:`~FrameViewSpaceWriterBase.set_poses` /
        :meth:`~FrameViewSpaceWriterBase.set_scales` write local-space values.

        Returns:
            A :class:`~isaaclab.sim.views.FrameViewLocalSpaceWriter` context manager.

        Raises:
            RuntimeError: On ``__enter__``, if another writer is already active
                on this view.

        Example:
            .. code-block:: python

                with view.xform_local_space_writer() as w:
                    w.set_poses(translations=t, orientations=o)
                    w.set_scales(scales=s)
        """
        return self._make_local_space_writer()

    @abc.abstractmethod
    def _make_world_space_writer(self) -> FrameViewWorldSpaceWriter:
        """Backend hook: return a fresh :class:`FrameViewWorldSpaceWriter` for this view."""
        ...

    @abc.abstractmethod
    def _make_local_space_writer(self) -> FrameViewLocalSpaceWriter:
        """Backend hook: return a fresh :class:`FrameViewLocalSpaceWriter` for this view."""
        ...

    def _assert_no_active_writer(self, method_name: str) -> None:
        """Raise :class:`RuntimeError` if a writer scope is currently active on this view."""
        if self._active_writer is not None:
            raise RuntimeError(
                f"{type(self).__name__}.{method_name}() is not allowed while a writer "
                f"scope is active ({type(self._active_writer).__name__}). Use the writer's "
                f"get_poses / get_scales inside the scope, or exit the scope first."
            )

    # ------------------------------------------------------------------
    # Public getters -- guarded; delegate to backend ``_*_impl`` hooks.
    # ------------------------------------------------------------------

    def get_world_poses(self, indices: wp.array | None = None) -> tuple[ProxyArray, ProxyArray]:
        """Get world-space positions and orientations for prims in the view.

        Args:
            indices: Subset of prims to query.  ``None`` means all prims.

        Returns:
            A tuple ``(positions, orientations)`` of :class:`~isaaclab.utils.warp.ProxyArray`
            wrappers.

        Raises:
            RuntimeError: If a writer scope is active on this view.
        """
        self._assert_no_active_writer("get_world_poses")
        return self._get_world_poses_impl(indices)

    def get_local_poses(self, indices: wp.array | None = None) -> tuple[ProxyArray, ProxyArray]:
        """Get local-space translations and orientations for prims in the view.

        Args:
            indices: Subset of prims to query.  ``None`` means all prims.

        Returns:
            A tuple ``(translations, orientations)`` of :class:`~isaaclab.utils.warp.ProxyArray`
            wrappers.

        Raises:
            RuntimeError: If a writer scope is active on this view.
        """
        self._assert_no_active_writer("get_local_poses")
        return self._get_local_poses_impl(indices)

    def get_local_scales(self, indices: wp.array | None = None) -> ProxyArray:
        """Get local-space scales for prims in the view.

        Args:
            indices: Subset of prims to query.  ``None`` means all prims.

        Returns:
            A :class:`~isaaclab.utils.warp.ProxyArray` of shape ``(M, 3)``.

        Raises:
            RuntimeError: If a writer scope is active on this view.
        """
        self._assert_no_active_writer("get_local_scales")
        return self._get_local_scales_impl(indices)

    def get_world_scales(self, indices: wp.array | None = None) -> ProxyArray:
        """Get world-space (composed) scales for prims in the view.

        Returns the effective scale in world space (``parent_scale * local_scale``).

        .. note::
            Scale extraction uses TRS (Translation-Rotation-Scale) decomposition,
            which assumes no shear/skew in the transform matrix.  If a prim's
            world transform contains shear, the extracted scale values will be
            approximate.

        Args:
            indices: Subset of prims to query.  ``None`` means all prims.

        Returns:
            A :class:`~isaaclab.utils.warp.ProxyArray` of shape ``(M, 3)``.

        Raises:
            RuntimeError: If a writer scope is active on this view.
        """
        self._assert_no_active_writer("get_world_scales")
        return self._get_world_scales_impl(indices)

    # ------------------------------------------------------------------
    # Backend hooks for the public getters above.
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def _get_world_poses_impl(self, indices: wp.array | None = None) -> tuple[ProxyArray, ProxyArray]:
        """Backend implementation of :meth:`get_world_poses`."""
        ...

    @abc.abstractmethod
    def _get_local_poses_impl(self, indices: wp.array | None = None) -> tuple[ProxyArray, ProxyArray]:
        """Backend implementation of :meth:`get_local_poses`."""
        ...

    @abc.abstractmethod
    def _get_local_scales_impl(self, indices: wp.array | None = None) -> ProxyArray:
        """Backend implementation of :meth:`get_local_scales`."""
        ...

    @abc.abstractmethod
    def _get_world_scales_impl(self, indices: wp.array | None = None) -> ProxyArray:
        """Backend implementation of :meth:`get_world_scales`."""
        ...

    # ------------------------------------------------------------------
    # Convenience pose/scale setters -- route through the writer scope.
    #
    # These are kept as first-class convenience APIs (not deprecated).  Each
    # call opens its own single-statement writer scope internally, so writing
    # poses and scales through two separate calls performs the opposite-space
    # recompute + synchronize twice.  For the best performance when updating
    # both poses and scales together, open one writer scope and issue both
    # writes inside it::
    #
    #     with view.xform_world_space_writer() as w:
    #         w.set_poses(...)
    #         w.set_scales(...)
    #
    # Prefer the writer scope in hot loops; prefer these helpers when code
    # simplicity matters more than shaving a redundant derive/sync.
    # ------------------------------------------------------------------

    def set_world_poses(
        self,
        positions: wp.array | None = None,
        orientations: wp.array | None = None,
        indices: wp.array | None = None,
    ) -> None:
        """Set world-space positions and/or orientations for prims in the view.

        This convenience method opens a single-statement writer scope
        internally.  To update poses and scales together without paying the
        opposite-space derive/sync twice, prefer
        ``with view.xform_world_space_writer() as w: w.set_poses(...); w.set_scales(...)``.

        Args:
            positions: World-space positions ``(M, 3)``. ``None`` leaves positions unchanged.
            orientations: World-space quaternions ``(M, 4)``. ``None`` leaves orientations unchanged.
            indices: Subset of prims to update.  ``None`` means all prims.
        """
        with self.xform_world_space_writer() as writer:
            writer.set_poses(positions, orientations, indices)

    def set_local_poses(
        self,
        translations: wp.array | None = None,
        orientations: wp.array | None = None,
        indices: wp.array | None = None,
    ) -> None:
        """Set local-space translations and/or orientations for prims in the view.

        This convenience method opens a single-statement writer scope
        internally.  To update poses and scales together without paying the
        opposite-space derive/sync twice, prefer
        ``with view.xform_local_space_writer() as w: w.set_poses(...); w.set_scales(...)``.

        Args:
            translations: Local-space translations ``(M, 3)``. ``None`` leaves translations unchanged.
            orientations: Local-space quaternions ``(M, 4)``. ``None`` leaves orientations unchanged.
            indices: Subset of prims to update.  ``None`` means all prims.
        """
        with self.xform_local_space_writer() as writer:
            writer.set_poses(translations, orientations, indices)

    # ------------------------------------------------------------------
    # Scale getter/setter convenience helpers.
    # ------------------------------------------------------------------

    def get_scales(self, indices: wp.array | None = None) -> ProxyArray:
        """Get scales for prims in the view.

        .. note::
            Prefer the explicit :meth:`get_local_scales` or
            :meth:`get_world_scales` when the space matters.  This method
            delegates to :meth:`_get_scales_impl`, which preserves each
            backend's legacy space (world for Fabric, local for USD).

        Args:
            indices: Subset of prims to query.  ``None`` means all prims.

        Returns:
            A ``ProxyArray`` of shape ``(M, 3)``.

        Raises:
            RuntimeError: If a writer scope is active on this view.
        """
        self._assert_no_active_writer("get_scales")
        return self._get_scales_impl(indices)

    def set_scales(self, scales: wp.array, indices: wp.array | None = None) -> None:
        """Set scales for prims in the view.

        This convenience method delegates to :meth:`_set_scales_impl`, which
        opens the backend's legacy space (world for Fabric, local for USD) and
        calls ``writer.set_scales``.  To update poses and scales together
        without paying the opposite-space derive/sync twice, prefer
        ``with view.xform_world_space_writer() as w: w.set_poses(...); w.set_scales(...)``
        (or :meth:`xform_local_space_writer`).

        Args:
            scales: Scales ``(M, 3)`` as ``wp.array``.
            indices: Subset of prims to update.  ``None`` means all prims.
        """
        self._set_scales_impl(scales, indices)

    @abc.abstractmethod
    def _get_scales_impl(self, indices: wp.array | None = None) -> ProxyArray:
        """Backend-specific implementation for :meth:`get_scales`."""
        ...

    @abc.abstractmethod
    def _set_scales_impl(self, scales: wp.array, indices: wp.array | None = None) -> None:
        """Backend-specific implementation for :meth:`set_scales`."""
        ...
