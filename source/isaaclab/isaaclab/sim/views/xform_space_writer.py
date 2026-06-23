# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Context-managed transform writers for :class:`~isaaclab.sim.views.BaseFrameView`.

This module defines the recommended write API for FrameView poses and scales:

.. code-block:: python

    with view.xform_world_space_writer() as writer:
        writer.set_poses(positions=p, orientations=o)
        writer.set_scales(scales=s)
        # ... any number of writes ...
    # On exit the writer derives the opposite-space matrices once,
    # synchronizes once, and restores any saved Fabric tracking state.

Only one writer may be active per view at a time.  While a writer scope is
active on a view, view-level getters (``view.get_world_poses``,
``view.get_local_poses``, ``view.get_world_scales``,
``view.get_local_scales``) raise :class:`RuntimeError` -- use the writer's own
:meth:`~FrameViewSpaceWriterBase.get_poses` / :meth:`~FrameViewSpaceWriterBase.get_scales`
inside the scope, or exit the scope first.

**Do not advance the simulation or render from inside a scope.**  The scope
runs as synchronous Python code, so no ``sim.step()`` / ``world.render()`` /
``SimulationApp.update()`` is allowed inside the ``with`` block.  Until the
scope exits, the backend's matrices may be mid-write (some prims updated,
others not; the opposite-space derive has not yet run) and rendering against
that state would read torn data.  Keep scopes short and step the
simulation outside them.
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING

import warp as wp

from isaaclab.utils.warp import ProxyArray

if TYPE_CHECKING:
    from .base_frame_view import BaseFrameView


class FrameViewSpaceWriterBase(abc.ABC):
    """Abstract context-managed writer for a single transform space.

    Subclasses are returned by :meth:`BaseFrameView.xform_world_space_writer` /
    :meth:`BaseFrameView.xform_local_space_writer`; they
    are not constructed directly.  The class is intentionally minimal -- the
    pose/scale semantics depend on the writer's space (world or local), which
    is conveyed by the concrete tag class :class:`FrameViewWorldSpaceWriter` or
    :class:`FrameViewLocalSpaceWriter`.

    The scope runs as synchronous Python code: no simulation step and no
    render tick can run while it is open, and the caller must not advance
    either from inside the ``with`` block.  See the module docstring for the
    full contract.
    """

    def __init__(self, view: BaseFrameView):
        self._view = view

    @abc.abstractmethod
    def set_poses(
        self,
        positions: wp.array | None = None,
        orientations: wp.array | None = None,
        indices: wp.array | None = None,
    ) -> None:
        """Set positions and/or orientations in this writer's space.

        Args:
            positions: Positions ``(M, 3)``. ``None`` leaves positions unchanged.
            orientations: Quaternions ``(M, 4)`` in ``(x, y, z, w)``.
                ``None`` leaves orientations unchanged.
            indices: Subset of prims to update.  ``None`` means all prims.
        """
        ...

    @abc.abstractmethod
    def set_scales(self, scales: wp.array, indices: wp.array | None = None) -> None:
        """Set scales in this writer's space.

        Args:
            scales: Scales ``(M, 3)`` as ``wp.array``.
            indices: Subset of prims to update.  ``None`` means all prims.
        """
        ...

    @abc.abstractmethod
    def get_poses(self, indices: wp.array | None = None) -> tuple[ProxyArray, ProxyArray]:
        """Return ``(positions, orientations)`` in this writer's space.

        Reflects any in-scope writes that have already been queued on the
        underlying device stream.
        """
        ...

    @abc.abstractmethod
    def get_scales(self, indices: wp.array | None = None) -> ProxyArray:
        """Return scales in this writer's space."""
        ...

    def __enter__(self) -> FrameViewSpaceWriterBase:
        if self._view._active_writer is not None:
            raise RuntimeError(
                f"{type(self._view).__name__} already has an active writer scope "
                f"({type(self._view._active_writer).__name__}). Exit the existing scope before "
                "opening a new one."
            )
        self._view._active_writer = self
        self._enter_impl()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        try:
            self._exit_impl(exc_type, exc_val, exc_tb)
        finally:
            self._view._active_writer = None

    def _enter_impl(self) -> None:
        """Backend hook called after the single-active-writer lock is claimed."""

    def _exit_impl(self, exc_type, exc_val, exc_tb) -> None:
        """Backend hook called before the single-active-writer lock is released."""


class FrameViewWorldSpaceWriter(FrameViewSpaceWriterBase):
    """Writer whose :meth:`set_poses` / :meth:`set_scales` write world-space values.

    On context exit the opposite-space (``local``) matrices are derived from
    the just-written world matrices in a single Warp kernel launch.
    """


class FrameViewLocalSpaceWriter(FrameViewSpaceWriterBase):
    """Writer whose :meth:`set_poses` / :meth:`set_scales` write local-space values.

    On context exit the opposite-space (``world``) matrices are derived from
    the just-written local matrices in a single Warp kernel launch.
    """
