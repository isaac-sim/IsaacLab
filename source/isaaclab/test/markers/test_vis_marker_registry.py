# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the visualization marker registry."""

from __future__ import annotations

import gc

import pytest

from isaaclab.markers.vis_marker_registry import VisMarkerRegistry

pytestmark = pytest.mark.unit


class _Owner:
    """Minimal debug-visualization owner; a real class so it can be weak-referenced."""

    def __init__(self) -> None:
        self.calls = 0

    def _debug_vis_callback(self, event) -> None:
        self.calls += 1


def test_add_and_clear_debug_vis_callback():
    """Registering returns an id, and clearing removes it and resets the owner's handle."""
    registry = VisMarkerRegistry()
    owner = _Owner()

    owner._debug_vis_handle = registry.add_debug_vis_callback(owner)
    assert isinstance(owner._debug_vis_handle, str)

    registry.dispatch_callbacks()
    assert owner.calls == 1

    registry.clear_debug_vis_callback(owner)
    assert owner._debug_vis_handle is None

    registry.dispatch_callbacks()
    assert owner.calls == 1


def test_dispatch_drops_callbacks_whose_owner_was_collected():
    """A collected owner must not abort dispatch for the callbacks that are still live.

    Callbacks hold a weak proxy, so an owner freed without deregistering leaves an entry
    that raises ``ReferenceError`` when invoked.
    """
    registry = VisMarkerRegistry()
    live = _Owner()
    dead = _Owner()

    registry.add_debug_vis_callback(live)
    registry.add_debug_vis_callback(dead)

    del dead
    gc.collect()

    registry.dispatch_callbacks()
    assert live.calls == 1

    # the stale entry is gone, so later dispatches keep working
    registry.dispatch_callbacks()
    assert live.calls == 2
