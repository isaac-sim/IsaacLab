# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Backend-independent tests for the deprecated FrameView scale helpers.

Exercises :meth:`~isaaclab.sim.views.BaseFrameView.get_scales` /
:meth:`~isaaclab.sim.views.BaseFrameView.set_scales` and the writer-scope lock
against a minimal in-memory view, so no simulation runtime is required.
"""

from __future__ import annotations

import warnings

import pytest
import warp as wp

from isaaclab.sim.views import base_frame_view as base_frame_view_module
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


@pytest.fixture(autouse=True)
def reset_warn_registry():
    """Clear the once-per-class warning cache so each test observes the first warning."""
    base_frame_view_module._DEPRECATION_WARNED.clear()
    yield
    base_frame_view_module._DEPRECATION_WARNED.clear()


def test_get_scales_emits_deprecation_warning_and_returns_warp_array():
    view = _StubFrameView()
    with pytest.warns(DeprecationWarning, match="get_scales") as caught:
        scales = view.get_scales()
    assert isinstance(scales, wp.array), f"get_scales() must return wp.array, got {type(scales).__name__}"
    assert "get_local_scales" in str(caught[0].message)
    assert "get_world_scales" in str(caught[0].message)


def test_set_scales_emits_deprecation_warning_and_writes_world_space():
    view = _StubFrameView()
    scales = wp.ones((COUNT, 3), dtype=wp.float32, device="cpu")
    with pytest.warns(DeprecationWarning, match="set_scales") as caught:
        view.set_scales(scales)
    assert view.scale_writes == ["_StubWorldWriter"]
    assert "xform_world_space_writer" in str(caught[0].message)


@pytest.mark.parametrize("method", ["get_scales", "set_scales"])
def test_deprecation_warning_is_emitted_once_per_class(method):
    view = _StubFrameView()
    scales = wp.ones((COUNT, 3), dtype=wp.float32, device="cpu")

    def call():
        if method == "get_scales":
            view.get_scales()
        else:
            view.set_scales(scales)

    with pytest.warns(DeprecationWarning):
        call()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        call()
        call()
    deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert deprecations == [], f"expected no repeat warning, got {[str(w.message) for w in deprecations]}"


def test_explicit_scale_getters_do_not_warn():
    view = _StubFrameView()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert isinstance(view.get_world_scales(), ProxyArray)
        assert isinstance(view.get_local_scales(), ProxyArray)
        with view.xform_world_space_writer() as writer:
            writer.set_scales(wp.ones((COUNT, 3), dtype=wp.float32, device="cpu"))
    deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert deprecations == [], f"explicit API must not warn, got {[str(w.message) for w in deprecations]}"


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
        for name in ("get_world_poses", "get_local_poses", "get_world_scales", "get_local_scales"):
            with pytest.raises(RuntimeError, match=name):
                getattr(view, name)()
        with pytest.warns(DeprecationWarning), pytest.raises(RuntimeError, match="get_scales"):
            view.get_scales()


def test_world_and_local_scale_reads_use_separate_buffers():
    """Reading one space must not overwrite a previously returned array of the other."""
    view = _StubFrameView()
    world = view.get_world_scales()
    local = view.get_local_scales()
    assert world.warp.ptr != local.warp.ptr, "world and local scale reads share one buffer"
