# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Backend-independent tests for the FrameView writer scope and scale getters.

Exercises the single-writer lock, the getter guard it installs, and the cached
scale buffers against a minimal in-memory view, so no simulation runtime is
required.
"""

from __future__ import annotations

import pytest
import warp as wp

from isaaclab.sim.views.base_frame_view import BaseFrameView
from isaaclab.sim.views.xform_space_writer import FrameViewLocalSpaceWriter, FrameViewWorldSpaceWriter
from isaaclab.utils.warp import ProxyArray

pytestmark = pytest.mark.unit

COUNT = 4


class _StubWriter:
    """Records scale writes and the space they were issued in."""

    def set_poses(self, positions=None, orientations=None, indices=None) -> None:
        pass

    def set_scales(self, scales, indices=None) -> None:
        self._view.scale_writes.append(type(self).__name__)

    def get_poses(self, indices=None):
        return self._view._get_world_poses_impl(indices)

    def get_scales(self, indices=None) -> ProxyArray:
        return self._view._get_world_scales_impl(indices)


class _StubWorldWriter(_StubWriter, FrameViewWorldSpaceWriter):
    pass


class _StubLocalWriter(_StubWriter, FrameViewLocalSpaceWriter):
    pass


class _StubFrameView(BaseFrameView):
    """Minimal concrete view: legacy scale space is world, like the Fabric backend."""

    def __init__(self, device: str = "cpu", enter_raises: bool = False):
        self._device_str = device
        self._enter_raises = enter_raises
        self.scale_writes: list[str] = []
        self._world_scales = ProxyArray(wp.zeros((COUNT, 3), dtype=wp.float32, device=device))
        self._local_scales = ProxyArray(wp.zeros((COUNT, 3), dtype=wp.float32, device=device))

    @property
    def count(self) -> int:
        return COUNT

    @property
    def device(self) -> str:
        return self._device_str

    def _make_world_space_writer(self) -> FrameViewWorldSpaceWriter:
        writer = _StubWorldWriter(self)
        if self._enter_raises:
            writer._enter_impl = _raise_backend_failure  # type: ignore[method-assign]
        return writer

    def _make_local_space_writer(self) -> FrameViewLocalSpaceWriter:
        return _StubLocalWriter(self)

    def _get_world_poses_impl(self, indices=None):
        return self._world_scales, self._world_scales

    def _get_local_poses_impl(self, indices=None):
        return self._local_scales, self._local_scales

    def _get_world_scales_impl(self, indices=None) -> ProxyArray:
        return self._world_scales

    def _get_local_scales_impl(self, indices=None) -> ProxyArray:
        return self._local_scales

    def _get_scales_impl(self, indices=None) -> ProxyArray:
        return self._get_world_scales_impl(indices)

    def _set_scales_impl(self, scales, indices=None) -> None:
        with self.xform_world_space_writer() as writer:
            writer.set_scales(scales, indices)


def _raise_backend_failure() -> None:
    raise RuntimeError("backend init failed")


def test_get_scales_returns_backend_legacy_space():
    """``get_scales`` resolves to the backend's legacy space; here that is world."""
    view = _StubFrameView()
    assert view.get_scales().warp.ptr == view.get_world_scales().warp.ptr


def test_set_scales_writes_backend_legacy_space():
    view = _StubFrameView()
    view.set_scales(wp.ones((COUNT, 3), dtype=wp.float32, device="cpu"))
    assert view.scale_writes == ["_StubWorldWriter"]


def test_writer_lock_released_when_enter_impl_raises():
    """A backend failure inside ``__enter__`` must not wedge the view forever."""
    view = _StubFrameView(enter_raises=True)
    with pytest.raises(RuntimeError, match="backend init failed"):
        with view.xform_world_space_writer():
            pass
    assert view._active_writer is None, "writer lock leaked after __enter__ raised"
    # The view must still be usable: getters unguarded, and a new scope can open.
    assert isinstance(view.get_world_scales(), ProxyArray)


def test_getters_are_guarded_while_writer_scope_is_active():
    view = _StubFrameView()
    with view.xform_world_space_writer():
        for name in ("get_world_poses", "get_local_poses", "get_world_scales", "get_local_scales", "get_scales"):
            with pytest.raises(RuntimeError, match=name):
                getattr(view, name)()


def test_world_and_local_scale_reads_use_separate_buffers():
    """Reading one space must not overwrite a previously returned array of the other."""
    view = _StubFrameView()
    world = view.get_world_scales()
    local = view.get_local_scales()
    assert world.warp.ptr != local.warp.ptr, "world and local scale reads share one buffer"
