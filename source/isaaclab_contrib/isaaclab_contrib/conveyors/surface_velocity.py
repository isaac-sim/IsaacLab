# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Import-light descriptions of prescribed conveyor surface velocities."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

_ENV_REGEX_NS = "{ENV_REGEX_NS}"


@dataclass(frozen=True, slots=True)
class SurfaceVelocitySpec:
    """Persistent intent for one collision surface with prescribed tangential velocity.

    Directions, surface normals, and the optional pivot point are expressed in the collision prim's
    local frame. Applied forces and reset state belong to the adapter.

    ``prim_path`` starts with Isaac Lab's ``{ENV_REGEX_NS}/`` placeholder. The adapter
    resolves it using the scene's clone template, preserving environment-major surface order.

    Args:
        prim_path: Replicated collision-prim template with concrete descendant names.
        velocity: Fixed signed centerline surface velocity [m/s].
        direction: Local travel direction, or local rotation axis for a curved belt.
        curved: Whether the velocity field rotates about ``pivot_point``.
        pivot_point: Local rotation center for a curved belt [m].
        radius: Centerline radius required for a curved belt [m].
        surface_normal: Local outward normal of the carrying surface.
        contact_threshold: Minimum contact-normal alignment accepted for traction.
        friction_coefficient: Coulomb limit for synthetic surface traction.
    """

    prim_path: str
    velocity: float = 0.0
    direction: tuple[float, float, float] = (1.0, 0.0, 0.0)
    curved: bool = False
    pivot_point: tuple[float, float, float] = (0.0, 0.0, 0.0)
    radius: float | None = None
    surface_normal: tuple[float, float, float] = (0.0, 0.0, 1.0)
    contact_threshold: float = 0.997
    friction_coefficient: float = 0.7

    def __post_init__(self) -> None:
        """Reject invalid authored geometry and contact parameters."""
        if not self.prim_path.startswith(f"{_ENV_REGEX_NS}/") or not all(
            part.isidentifier() for part in self.prim_path[len(_ENV_REGEX_NS) + 1 :].split("/")
        ):
            raise ValueError(f"Surface prim_path must start with '{_ENV_REGEX_NS}/' and name concrete prims.")
        for name, vector in (
            ("direction", self.direction),
            ("pivot_point", self.pivot_point),
            ("surface_normal", self.surface_normal),
        ):
            if len(vector) != 3 or not all(math.isfinite(value) for value in vector):
                raise ValueError(f"Surface {name} must contain three finite values.")
            if name != "pivot_point" and math.hypot(*vector) <= 1.0e-8:
                raise ValueError(f"Surface {name} must be non-zero.")
        if not math.isfinite(self.velocity):
            raise ValueError("Surface velocity must be finite.")
        if not 0.0 <= self.contact_threshold <= 1.0:
            raise ValueError(f"Surface contact_threshold must be in [0, 1], got {self.contact_threshold!r}.")
        if not math.isfinite(self.friction_coefficient) or self.friction_coefficient < 0.0:
            raise ValueError("Surface friction_coefficient must be finite and non-negative.")
        if self.curved and self.radius is None:
            raise ValueError("Curved surfaces require a positive radius.")
        if self.radius is not None and (not math.isfinite(self.radius) or self.radius <= 0.0):
            raise ValueError("Surface radius must be finite and positive.")


def resolve_surface_velocity_paths(
    num_envs: int,
    surface_specs: Sequence[SurfaceVelocitySpec],
    env_path_format: str = "/World/envs/env_{}",
) -> tuple[str, ...]:
    """Resolve conveyor templates to exact environment-major prim paths.

    Args:
        num_envs: Number of replicated environments.
        surface_specs: Within-environment surface descriptions.
        env_path_format: Exact environment path format containing one ``{}`` field.

    Returns:
        Exact paths ordered by environment, then by ``surface_specs`` order.

    Raises:
        ValueError: If surface ownership is ambiguous or the clone template is invalid.
    """
    if num_envs < 1:
        raise ValueError("Conveyor num_envs must be positive.")
    if not surface_specs:
        raise ValueError("At least one SurfaceVelocitySpec is required.")
    templates = tuple(spec.prim_path for spec in surface_specs)
    ordered_templates = sorted(templates)
    if any(
        right == left or right.startswith(f"{left}/") for left, right in zip(ordered_templates, ordered_templates[1:])
    ):
        raise ValueError("Conveyor prim paths must be unique and must not be ancestors of one another.")
    if env_path_format.count("{}") != 1:
        raise ValueError("Conveyor env_path_format must contain exactly one '{}'.")
    env_paths = tuple(env_path_format.format(env_id) for env_id in range(num_envs))
    if any(
        not path.startswith("/") or not all(part.isidentifier() for part in path[1:].split("/")) for path in env_paths
    ):
        raise ValueError("Conveyor env_path_format must produce concrete absolute prim paths.")
    return tuple(template.replace(_ENV_REGEX_NS, env_path) for env_path in env_paths for template in templates)
