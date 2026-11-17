# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for prescribed conveyor surface motion."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

from ..utils import configclass

if TYPE_CHECKING:
    from .conveyor import Conveyor


@configclass
class ConveyorSurfaceCfg:
    """Describe one collision surface's velocity profile in its local frame.

    ``prim_path`` starts with ``{ENV_REGEX_NS}/`` and names concrete descendants.
    Straight profiles translate along ``direction``. Curved profiles rotate about
    ``pivot_point`` and use ``direction`` as their rotation axis.
    """

    prim_path: str = MISSING
    """Collision-prim template within each replicated environment."""

    velocity: float = 0.0
    """Signed centerline surface velocity [m/s] at speed scale one."""

    direction: tuple[float, float, float] = (1.0, 0.0, 0.0)
    """Local travel direction, or rotation axis for a curved profile."""

    curved: bool = False
    """Whether the velocity field rotates about :attr:`pivot_point`."""

    pivot_point: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Local rotation center [m] for a curved profile."""

    radius: float | None = None
    """Centerline radius [m], required for a curved profile."""

    surface_normal: tuple[float, float, float] = (0.0, 0.0, 1.0)
    """Local outward normal of the carrying surface."""

    contact_threshold: float = 0.997
    """Minimum contact-normal alignment accepted for traction."""

    friction_coefficient: float = 0.7
    """Coulomb limit for conveyor traction."""

    optional: bool = False
    """Permit this surface to be absent from some scene variants."""

    def validate_config(self) -> None:
        """Reject invalid authored geometry and contact parameters."""
        namespace = "{ENV_REGEX_NS}"
        if self.prim_path is not MISSING and (
            not self.prim_path.startswith(f"{namespace}/")
            or not all(part.isidentifier() for part in self.prim_path[len(namespace) + 1 :].split("/"))
        ):
            raise ValueError(f"Surface prim_path must start with '{namespace}/' and name concrete prims.")
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


@configclass
class ConveyorCfg:
    """Configuration for a batched conveyor controller.

    The surface profiles define the authored velocities. Runtime commands scale
    every profile within an environment together, preserving their relative motion.
    Currently implemented by the Newton MJWarp backend.
    """

    class_type: type[Conveyor] | str = "{DIR}.conveyor:Conveyor"
    """Controller implementation selected through the active physics backend."""

    surface_specs: Sequence[ConveyorSurfaceCfg] = MISSING
    """Ordered collision-surface profiles within each environment."""

    body_pattern: str = MISSING
    """Regular expression selecting transported body paths."""

    body_count_per_env: int = MISSING
    """Expected transported-body count in each environment."""

    startup_duration_s: float = 1.0
    """Initial traction-ramp duration [s]."""

    def validate_config(self) -> None:
        """Validate transported-body counts, startup duration, and distinct surface profiles."""
        if not isinstance(self.body_count_per_env, int) or self.body_count_per_env < 1:
            raise ValueError("Conveyor body_count_per_env must be a positive integer.")
        if not math.isfinite(self.startup_duration_s) or self.startup_duration_s <= 0.0:
            raise ValueError("Conveyor startup_duration_s must be finite and positive.")
        paths = sorted(surface.prim_path for surface in self.surface_specs)
        if any(right == left or right.startswith(f"{left}/") for left, right in zip(paths, paths[1:])):
            raise ValueError("Conveyor prim paths must be unique and must not be ancestors of one another.")
