# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Backend-neutral surface-velocity descriptions and control interface."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

_ENV_REGEX_NS = "{ENV_REGEX_NS}"


def _validate_prim_path(value: str) -> None:
    """Validate one exact USD prim path or supported replicated-path template."""
    if not isinstance(value, str) or not value:
        raise ValueError("Surface prim_path must be a non-empty string.")
    if value.startswith(f"{_ENV_REGEX_NS}/"):
        path = value[len(_ENV_REGEX_NS) :]
    elif value.startswith("/"):
        path = value
    else:
        raise ValueError(f"Surface prim_path must be absolute or start with '{_ENV_REGEX_NS}/', got {value!r}.")
    if "{" in path or "}" in path:
        raise ValueError(f"Surface prim_path supports only a leading '{_ENV_REGEX_NS}' placeholder, got {value!r}.")
    components = path[1:].split("/")
    if not components or any(not component or component in {".", ".."} for component in components):
        raise ValueError(f"Surface prim_path must identify a concrete prim without empty components, got {value!r}.")
    if any(any(character.isspace() for character in component) for component in components):
        raise ValueError(f"Surface prim_path components must not contain whitespace, got {value!r}.")


def _validate_scalar(name: str, value: Any) -> float:
    """Return one finite float with consistent validation errors."""
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"Surface {name} must be finite, got {value!r}.") from exc
    if not math.isfinite(result):
        raise ValueError(f"Surface {name} must be finite, got {value!r}.")
    return result


def _validate_vector(name: str, value: tuple[float, ...], length: int, *, nonzero: bool = False) -> tuple[float, ...]:
    """Return one finite vector with a stable tuple representation."""
    try:
        result = tuple(float(component) for component in value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Surface {name} must contain {length} finite values, got {value!r}.") from exc
    if len(result) != length or not all(math.isfinite(component) for component in result):
        raise ValueError(f"Surface {name} must contain {length} finite values, got {value!r}.")
    if nonzero and math.sqrt(sum(component * component for component in result)) <= 1.0e-8:
        raise ValueError(f"Surface {name} must be non-zero, got {value!r}.")
    return result


@dataclass(frozen=True, slots=True)
class SurfaceVelocitySpec:
    """Persistent intent for one collision surface with prescribed tangential velocity.

    The fields follow the authored conveyor model proposed for Isaac Sim while remaining independent of
    Kit, OpenUSD, and any physics backend. Directions, surface normals, and the optional pivot point are
    expressed in the collision prim's local frame. Runtime state such as encoder positions and applied
    forces intentionally does not belong in this description.

    ``prim_path`` may contain Isaac Lab's ``{ENV_REGEX_NS}`` placeholder. A backend resolves that template
    to every replicated collision prim while preserving one deterministic belt index per environment. An
    absolute path addresses one unreplicated surface; replicated environments should use the placeholder.

    Args:
        prim_path: Collision prim path or replicated Isaac Lab prim-path template.
        velocity: Initial signed centerline surface velocity [m/s].
        enabled: Whether the surface initially transports contacting bodies.
        direction: Local travel direction, or local rotation axis for a curved belt.
        curved: Whether the velocity field rotates about ``pivot_point``.
        pivot_point: Local rotation center for a curved belt [m]. An OpenUSD adapter may map this point to the
            world transform of a prim targeted by the proposed schema's pivot relationship; it is not copied
            into a schema attribute.
        radius: Optional centerline radius used by backends that cannot derive it from geometry [m].
        surface_normal: Local outward normal of the carrying surface.
        contact_threshold: Minimum contact-normal alignment accepted for traction.
        friction_coefficient: Coulomb limit for synthetic surface traction.
    """

    prim_path: str
    velocity: float = 0.0
    enabled: bool = True
    direction: tuple[float, float, float] = (1.0, 0.0, 0.0)
    curved: bool = False
    pivot_point: tuple[float, float, float] = (0.0, 0.0, 0.0)
    radius: float | None = None
    surface_normal: tuple[float, float, float] = (0.0, 0.0, 1.0)
    contact_threshold: float = 0.997
    friction_coefficient: float = 0.7

    def __post_init__(self) -> None:
        """Normalize immutable vectors and validate authored values."""
        _validate_prim_path(self.prim_path)
        for name in ("enabled", "curved"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"Surface {name} must be a bool, got {getattr(self, name)!r}.")

        velocity = _validate_scalar("velocity", self.velocity)
        contact_threshold = _validate_scalar("contact_threshold", self.contact_threshold)
        friction_coefficient = _validate_scalar("friction_coefficient", self.friction_coefficient)
        if not 0.0 <= contact_threshold <= 1.0:
            raise ValueError(f"Surface contact_threshold must be in [0, 1], got {self.contact_threshold!r}.")
        if friction_coefficient < 0.0:
            raise ValueError(
                f"Surface friction_coefficient must be finite and non-negative, got {self.friction_coefficient!r}."
            )
        radius = None if self.radius is None else _validate_scalar("radius", self.radius)
        if radius is not None and radius <= 0.0:
            raise ValueError(f"Surface radius must be finite and positive when provided, got {self.radius!r}.")

        object.__setattr__(self, "velocity", velocity)
        object.__setattr__(self, "direction", _validate_vector("direction", self.direction, 3, nonzero=True))
        object.__setattr__(self, "pivot_point", _validate_vector("pivot_point", self.pivot_point, 3))
        object.__setattr__(
            self, "surface_normal", _validate_vector("surface_normal", self.surface_normal, 3, nonzero=True)
        )
        object.__setattr__(self, "radius", radius)
        object.__setattr__(self, "contact_threshold", contact_threshold)
        object.__setattr__(self, "friction_coefficient", friction_coefficient)


@runtime_checkable
class SurfaceVelocityView(Protocol):
    """Common tensorized control contract implemented by physics backends."""

    @property
    def prim_paths(self) -> tuple[str, ...]:
        """Resolved collision prim paths in stable surface-index order."""
        ...

    @property
    def num_surfaces(self) -> int:
        """Number of resolved moving surfaces."""
        ...

    @property
    def count(self) -> int:
        """Alias for :attr:`num_surfaces`, matching tensor-view naming."""
        ...

    def set_velocities(self, velocities: Any, indices: Any = None) -> None:
        """Set signed surface velocities [m/s] for selected surfaces."""
        ...

    def get_velocities(self, indices: Any = None, clone: bool = True) -> Any:
        """Return effective surface velocities [m/s] for selected surfaces."""
        ...

    def get_commanded_velocities(self, indices: Any = None, clone: bool = True) -> Any:
        """Return commanded surface velocities [m/s] before the enabled mask."""
        ...

    def set_enabled(self, flags: Any, indices: Any = None) -> None:
        """Enable or disable selected surfaces without discarding their commands."""
        ...

    def get_enabled(self, indices: Any = None, clone: bool = True) -> Any:
        """Return integer enabled flags for selected surfaces."""
        ...

    def get_encoder_positions(self, indices: Any = None, clone: bool = True) -> Any:
        """Return integrated surface travel [m] for selected surfaces."""
        ...

    def reset(self, env_ids: Any = None) -> None:
        """Clear runtime state for selected replicated environments."""
        ...

    def close(self) -> None:
        """Release backend resources and callbacks."""
        ...
