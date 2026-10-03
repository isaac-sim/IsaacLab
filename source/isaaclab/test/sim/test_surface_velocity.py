# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Focused tests for backend-neutral surface-velocity descriptions."""

from __future__ import annotations

import pytest

from isaaclab.physics import SurfaceVelocitySpec


def test_surface_velocity_spec_normalizes_vectors() -> None:
    """List inputs become immutable vectors before either backend consumes them."""
    spec = SurfaceVelocitySpec(
        prim_path="{ENV_REGEX_NS}/Belt", direction=[3, 4, 0], pivot_point=[1, 2, 3], surface_normal=[0, 0, 1]
    )
    assert spec.direction == (3.0, 4.0, 0.0)
    assert spec.pivot_point == (1.0, 2.0, 3.0)
    assert spec.surface_normal == (0.0, 0.0, 1.0)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"prim_path": "relative/Belt"}, "prim_path"),
        ({"prim_path": "/"}, "prim_path"),
        ({"prim_path": "/World/{ENV_REGEX_NS}/Belt"}, "prim_path"),
        ({"prim_path": "/World/Belt", "velocity": None}, "velocity"),
        ({"prim_path": "/World/Belt", "velocity": float("nan")}, "velocity"),
        ({"prim_path": "/World/Belt", "enabled": 1}, "enabled"),
        ({"prim_path": "/World/Belt", "curved": "yes"}, "curved"),
        ({"prim_path": "/World/Belt", "direction": (0.0, 0.0, 0.0)}, "direction"),
        ({"prim_path": "/World/Belt", "surface_normal": (0.0, 0.0)}, "surface_normal"),
        ({"prim_path": "/World/Belt", "radius": 0.0}, "radius"),
        ({"prim_path": "/World/Belt", "contact_threshold": 1.1}, "contact_threshold"),
        ({"prim_path": "/World/Belt", "friction_coefficient": -0.1}, "friction_coefficient"),
    ],
)
def test_surface_velocity_spec_rejects_invalid_authored_values(kwargs: dict, message: str) -> None:
    """Invalid persistent intent fails before any physics lifecycle is registered."""
    with pytest.raises(ValueError, match=message):
        SurfaceVelocitySpec(**kwargs)
