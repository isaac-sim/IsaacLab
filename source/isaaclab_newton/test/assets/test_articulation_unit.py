# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Kit-less unit tests for Newton articulation logic."""

from types import SimpleNamespace

from isaaclab_newton.assets import Articulation


def test_num_shapes_per_body_follows_public_body_order() -> None:
    """Align Newton shape counts with the public body-name axis."""

    class _ShapeCountSurface:
        backend_num_shapes_per_body = Articulation.backend_num_shapes_per_body
        num_shapes_per_body = Articulation.num_shapes_per_body

    articulation = _ShapeCountSurface()
    articulation._num_shapes_per_body_backend = None
    articulation._root_view = SimpleNamespace(
        body_shapes=((), (object(), object()), (object(), object(), object())),
    )
    articulation.body_ordering = SimpleNamespace(
        user_to_backend_indices=(2, 0, 1),
    )

    assert articulation.num_shapes_per_body == [3, 0, 2]
