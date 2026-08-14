# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the backend-neutral conveyor belt contract."""

from __future__ import annotations

import pytest

from isaaclab.physics import ConveyorBeltSpec


def test_conveyor_belt_spec_preserves_authored_semantics() -> None:
    """The shared description carries schema-aligned fields without backend imports."""
    spec = ConveyorBeltSpec(
        prim_path="{ENV_REGEX_NS}/Belt/Curve",
        velocity=-0.35,
        enabled=False,
        direction=(0, 0, -1),
        curved=True,
        pivot_point=(0.58, 0.51, 0.0),
        radius=0.24,
        surface_normal=(0, 0, 1),
        contact_threshold=0.997,
        friction_coefficient=0.5,
        animate_texture=True,
        animate_direction=(1, 0),
        animate_scale=0.5,
    )

    assert spec.prim_path == "{ENV_REGEX_NS}/Belt/Curve"
    assert spec.velocity == -0.35
    assert spec.enabled is False
    assert spec.direction == (0.0, 0.0, -1.0)
    assert spec.curved is True
    assert spec.pivot_point == (0.58, 0.51, 0.0)
    assert spec.radius == 0.24
    assert spec.surface_normal == (0.0, 0.0, 1.0)
    assert spec.contact_threshold == 0.997
    assert spec.friction_coefficient == 0.5
    assert spec.animate_texture is True
    assert spec.animate_direction == (1.0, 0.0)
    assert spec.animate_scale == 0.5


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"prim_path": "relative/Belt"}, "prim_path"),
        ({"prim_path": "/"}, "prim_path"),
        ({"prim_path": "/World//Belt"}, "prim_path"),
        ({"prim_path": "/World/Belt/"}, "prim_path"),
        ({"prim_path": "{ENV_REGEX_NS}/"}, "prim_path"),
        ({"prim_path": "{UNKNOWN_NS}/Belt"}, "prim_path"),
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
        ({"prim_path": "/World/Belt", "animate_scale": -1.0}, "animate_scale"),
        (
            {"prim_path": "/World/Belt", "animate_texture": True, "animate_direction": (0.0, 0.0)},
            "animate_direction",
        ),
    ],
)
def test_conveyor_belt_spec_rejects_invalid_authored_values(kwargs: dict, message: str) -> None:
    """Invalid persistent intent fails before any physics lifecycle is registered."""
    with pytest.raises(ValueError, match=message):
        ConveyorBeltSpec(**kwargs)
