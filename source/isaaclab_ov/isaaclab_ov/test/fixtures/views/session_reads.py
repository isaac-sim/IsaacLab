# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Helpers for tests that compare OvPhysX session reads with their TensorBindings on a real simulation."""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from unittest.mock import patch

import warp as wp

from isaaclab_ov.sim.views.ovphysx_session_reads import OvPhysxSessionReads
from isaaclab_ov.sim.views.ovphysx_view import OvPhysxView

REPLAY_ONLY = "read_into did not replay its captured session read"
"""Failure of a binding read or a new capture where only the captured plan may serve the read."""


def binding_values(view: OvPhysxView, tensor_type: int) -> wp.array:
    """Read ``tensor_type`` straight from its TensorBinding: the reference a session read must match.

    Shape and device come from the binding itself, since ``get_attribute`` would run (and capture) a session read.
    """
    binding = view.binding_for(tensor_type)
    values = wp.zeros(tuple(binding.shape), dtype=wp.float32, device=view._native_device(view._resolve(tensor_type)))
    binding.read(view._read_view(values, binding))
    return values


@contextlib.contextmanager
def replays_only(view: OvPhysxView, tensor_type: int) -> Iterator[None]:
    """Fail unless the one ``read_into`` inside replays a plan captured earlier: no binding read, no new capture.

    Without it, a read that fell back to the binding would match the binding reference trivially.
    """
    with (
        patch.object(type(view.binding_for(tensor_type)), "read", side_effect=AssertionError(REPLAY_ONLY)),
        patch.object(OvPhysxSessionReads, "_capture", side_effect=AssertionError(REPLAY_ONLY)),
        patch.object(wp, "capture_launch", wraps=wp.capture_launch) as launch,
    ):
        yield
    assert launch.call_count == 1, REPLAY_ONLY
