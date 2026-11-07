# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Focused tests for import-light surface-velocity descriptions."""

from __future__ import annotations

import pytest

from isaaclab_contrib.conveyors.surface_velocity import SurfaceVelocitySpec, resolve_surface_velocity_paths


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"prim_path": "relative/Belt"}, "prim_path"),
        ({"prim_path": "{ENV_REGEX_NS}/Belt.*"}, "prim_path"),
        ({"velocity": float("nan")}, "velocity"),
        ({"direction": (0.0, 0.0, 0.0)}, "direction"),
        ({"surface_normal": (0.0, 0.0)}, "surface_normal"),
        ({"pivot_point": (0.0, float("inf"), 0.0)}, "pivot_point"),
        ({"curved": True}, "radius"),
        ({"radius": 0.0}, "radius"),
        ({"contact_threshold": 1.1}, "contact_threshold"),
        ({"friction_coefficient": -0.1}, "friction_coefficient"),
    ],
)
def test_surface_velocity_spec_rejects_invalid_authored_values(kwargs: dict, message: str) -> None:
    """Invalid persistent intent fails before any physics lifecycle is registered."""
    kwargs = {"prim_path": "{ENV_REGEX_NS}/Belt", **kwargs}
    with pytest.raises(ValueError, match=message):
        SurfaceVelocitySpec(**kwargs)


def test_paths_are_resolved_in_environment_major_order() -> None:
    """Runtime rows stay deterministic across stage discovery ordering."""
    specs = (SurfaceVelocitySpec("{ENV_REGEX_NS}/Nested/BeltA"), SurfaceVelocitySpec("{ENV_REGEX_NS}/BeltB"))

    paths = resolve_surface_velocity_paths(2, specs, "/Factory/cell_{}")

    assert paths == (
        "/Factory/cell_0/Nested/BeltA",
        "/Factory/cell_0/BeltB",
        "/Factory/cell_1/Nested/BeltA",
        "/Factory/cell_1/BeltB",
    )


@pytest.mark.parametrize("paths", [("Belt", "Belt"), ("Belt/Child", "Belt")])
def test_paths_reject_ambiguous_surface_ownership(paths: tuple[str, str]) -> None:
    """One collider cannot be assigned to duplicate or overlapping conveyor roots."""
    specs = tuple(SurfaceVelocitySpec(f"{{ENV_REGEX_NS}}/{path}") for path in paths)
    with pytest.raises(ValueError, match="unique.*ancestors"):
        resolve_surface_velocity_paths(1, specs)
