# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for :class:`NewtonVisualizer` viewer release.

The RTX viewer owns GPU resources that it releases in a fixed order when
``close()`` is called. Both paths that give up the viewer -- ``close()`` and
the ``step()`` handler that permanently disables it after an unrecoverable
failure -- must go through :meth:`NewtonVisualizer._release_viewer` so that
ordering is honoured instead of being left to the garbage collector. The GL
viewer must keep its established reference-drop behavior because closing its
renderer prevents reliable in-process recreation.

These tests assert the behavior (the RTX viewer's ``close()`` runs before the
reference is dropped, while the GL viewer is not closed) rather than the
absence of a backend log message. The message is emitted for only one of
several valid finalization orders, so asserting on it would pass against
unfixed code most of the time.

They also pin the error semantics, which differ by caller.  ``_release_viewer``
propagates a teardown failure while still clearing the reference.  ``close()``
lets it propagate to ``SimulationContext``, which already logs it, but finishes
its own cleanup first.  ``step()`` contains it, because that handler exists so
an unusable viewer disables itself instead of aborting training.
"""

from __future__ import annotations

import isaaclab_visualizers.newton.newton_visualizer as newton_visualizer
import pytest
from isaaclab_visualizers.newton.newton_visualizer import NewtonVisualizer

pytestmark = [pytest.mark.unit]


class _SpyRTXViewer(newton_visualizer.NewtonViewerRTX):
    """RTX viewer double that records how and when it was closed."""

    def __init__(self, raises: bool = False) -> None:
        self.close_calls = 0
        self.referenced_by_owner_at_close: list[bool] = []
        self.referenced_by_picking_at_close: list[bool] = []
        self.apply_forces_calls = 0
        self.owner: NewtonVisualizer | None = None
        self._raises = raises

    def close(self) -> None:
        self.close_calls += 1
        # Record whether the visualizer still pointed at us while we were being
        # closed.  The reference must outlive the teardown call.
        self.referenced_by_owner_at_close.append(getattr(self.owner, "_viewer", None) is self)
        binding = getattr(self.owner, "_viewer_picking_binding", None)
        self.referenced_by_picking_at_close.append(getattr(binding, "_viewer", None) is self)
        if self._raises:
            raise RuntimeError("Failed to create window")

    def apply_forces(self, _state: object) -> None:
        """Record a picking callback reaching this viewer."""
        self.apply_forces_calls += 1


class _SpyGLViewer(newton_visualizer.NewtonViewerGL):
    """GL viewer double that records whether ``close()`` was called."""

    def __init__(self) -> None:
        self.close_calls = 0
        self.owner: NewtonVisualizer | None = None

    def close(self) -> None:
        self.close_calls += 1


def _make_visualizer(viewer: _SpyRTXViewer | _SpyGLViewer | None) -> NewtonVisualizer:
    """Build the minimal visualizer state that the release paths read.

    ``__init__`` is bypassed deliberately: a real visualizer requires a Newton
    model, a scene data provider and a GPU, none of which this behaviour
    depends on.
    """
    visualizer = object.__new__(NewtonVisualizer)
    visualizer._is_closed = False
    visualizer._picking_enabled = False
    visualizer._viewer_picking_binding = NewtonVisualizer._ViewerPickingBinding()
    visualizer._viewer = viewer
    visualizer._camera_sensor = None
    visualizer._camera_is_owned = False
    if viewer is not None:
        viewer.owner = visualizer
    return visualizer


def test_release_viewer_closes_before_clearing_reference() -> None:
    """The viewer must be closed while the visualizer still references it."""
    viewer = _SpyRTXViewer()
    visualizer = _make_visualizer(viewer)

    visualizer._release_viewer()

    assert viewer.close_calls == 1
    assert viewer.referenced_by_owner_at_close == [True]
    assert visualizer._viewer is None


def test_release_viewer_propagates_failure_and_still_clears_reference() -> None:
    """A teardown failure must reach the caller, but must not retain the viewer."""
    viewer = _SpyRTXViewer(raises=True)
    visualizer = _make_visualizer(viewer)

    with pytest.raises(RuntimeError, match="Failed to create window"):
        visualizer._release_viewer()

    assert viewer.close_calls == 1
    assert visualizer._viewer is None


def test_release_viewer_without_viewer_is_a_no_op() -> None:
    """Releasing when no viewer is held must be harmless."""
    visualizer = _make_visualizer(None)

    visualizer._release_viewer()

    assert visualizer._viewer is None


def test_release_viewer_is_idempotent() -> None:
    """Releasing twice must not close the viewer twice."""
    viewer = _SpyRTXViewer()
    visualizer = _make_visualizer(viewer)

    visualizer._release_viewer()
    visualizer._release_viewer()

    assert viewer.close_calls == 1


def test_release_viewer_does_not_close_gl_viewer() -> None:
    """GL teardown must not prevent another viewer from starting in the same process."""
    viewer = _SpyGLViewer()
    visualizer = _make_visualizer(viewer)

    visualizer._release_viewer()

    assert viewer.close_calls == 0
    assert visualizer._viewer is None


def test_close_releases_the_viewer() -> None:
    """``close()`` must release the viewer through the shared path."""
    viewer = _SpyRTXViewer()
    visualizer = _make_visualizer(viewer)

    visualizer.close()

    assert viewer.close_calls == 1
    assert viewer.referenced_by_owner_at_close == [True]
    assert visualizer._viewer is None
    assert visualizer._is_closed is True


def test_close_is_idempotent() -> None:
    """A second ``close()`` must not close the viewer again."""
    viewer = _SpyRTXViewer()
    visualizer = _make_visualizer(viewer)

    visualizer.close()
    visualizer.close()

    assert viewer.close_calls == 1


def test_close_completes_cleanup_when_viewer_teardown_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failing viewer must not strand the owned camera or the closed flag.

    ``SimulationContext`` already logs an exception raised by ``close()``, so
    it is allowed to propagate -- but the rest of the teardown still has to
    run, otherwise a viewer failure silently leaks the generated camera prims.
    """
    evicted: list[object] = []
    removed: list[object] = []
    monkeypatch.setattr(newton_visualizer, "evict_visualizer_camera", evicted.append, raising=False)
    monkeypatch.setattr(newton_visualizer, "remove_generated_prims", removed.append, raising=False)

    viewer = _SpyRTXViewer(raises=True)
    visualizer = _make_visualizer(viewer)
    visualizer._camera_sensor = object()
    visualizer._camera_is_owned = True
    visualizer._streaming_camera_key = "camera-key"
    visualizer._generated_camera_prim_paths = ["/World/generated"]

    with pytest.raises(RuntimeError, match="Failed to create window"):
        visualizer.close()

    assert visualizer._viewer is None
    assert visualizer._camera_sensor is None
    assert visualizer._is_closed is True
    assert evicted == ["camera-key"]
    assert removed == [["/World/generated"]]


def _arm_for_step_failure(visualizer: NewtonVisualizer, viewer: _SpyRTXViewer) -> None:
    """Drive ``step()`` far enough to reach its viewer-failure handler."""
    visualizer._is_initialized = True
    visualizer._runtime_headless = False
    visualizer._disable_viewer_on_step_exception = True
    visualizer._sim_time = 0.0
    visualizer._step_counter = 0
    visualizer._state = None
    visualizer._scene_data_provider = None
    visualizer._update_frequency = 1
    viewer._update_frequency = 1

    def _unrecoverable() -> bool:
        raise RuntimeError("Failed to create window")

    viewer.is_paused = _unrecoverable  # type: ignore[method-assign]


def test_step_failure_releases_the_viewer(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unrecoverable viewer failure during ``step()`` must release the viewer.

    ``NewtonRTXVisualizer`` sets ``_disable_viewer_on_step_exception`` so the
    viewer is given up after the first failure -- for example when OVRTX cannot
    create its window.  That path must close the viewer rather than only
    dropping the reference to it.
    """
    viewer = _SpyRTXViewer()
    visualizer = _make_visualizer(viewer)
    visualizer._picking_enabled = True
    visualizer._viewer_picking_binding.bind(viewer)  # type: ignore[arg-type]
    _arm_for_step_failure(visualizer, viewer)
    monkeypatch.setattr(newton_visualizer.NewtonManager, "get_num_envs", staticmethod(lambda: 1), raising=False)

    NewtonVisualizer.step(visualizer, dt=0.01)  # must not raise

    assert viewer.close_calls == 1
    assert viewer.referenced_by_owner_at_close == [True]
    assert viewer.referenced_by_picking_at_close == [False]
    assert visualizer._viewer is None

    # The callback remains registered with NewtonManager for CUDA graph
    # stability, but it must be inert after the viewer is released.
    visualizer._viewer_picking_binding.apply(None)  # type: ignore[arg-type]
    assert viewer.apply_forces_calls == 0


def test_step_contains_a_failing_viewer_teardown(monkeypatch: pytest.MonkeyPatch) -> None:
    """A viewer that fails to close must not abort training from ``step()``.

    This is the whole purpose of the ``_disable_viewer_on_step_exception``
    handler: the viewer is already known to be broken, so its teardown failure
    has to be contained rather than replacing the original failure and
    propagating out of the simulation loop.
    """
    viewer = _SpyRTXViewer(raises=True)
    visualizer = _make_visualizer(viewer)
    _arm_for_step_failure(visualizer, viewer)
    monkeypatch.setattr(newton_visualizer.NewtonManager, "get_num_envs", staticmethod(lambda: 1), raising=False)

    NewtonVisualizer.step(visualizer, dt=0.01)  # must not raise

    assert viewer.close_calls == 1
    assert visualizer._viewer is None
