# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CPU-only native PhysX surface-velocity conveyors for the conveyor-Franka task.

The module deliberately keeps PhysX schema imports behind authoring and binding calls.  Its
geometry conversion and host-side control state therefore remain usable in import-light tests
and tooling that do not launch Kit.

The task configuration rejects GPU dynamics: the supported native ``PhysxSurfaceVelocityAPI``
path can lose conveyor contacts there. The force-driven Newton adapter is the scalable GPU path.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from isaaclab.physics import ConveyorBeltSpec

_ENV_REGEX_NS = "{ENV_REGEX_NS}"


@dataclass(frozen=True, slots=True)
class PhysxSurfaceVelocityTwist:
    """Local PhysX surface twist.

    Attributes:
        linear_velocity: Local linear surface velocity [m/s].
        angular_velocity_deg: Local angular surface velocity [deg/s].
    """

    linear_velocity: tuple[float, float, float]
    angular_velocity_deg: tuple[float, float, float]


def compute_physx_surface_velocity_twist(
    spec: ConveyorBeltSpec, velocity: float | None = None
) -> PhysxSurfaceVelocityTwist:
    """Convert one backend-neutral belt command to a local PhysX surface twist.

    Straight belts map their signed speed to a normalized linear direction.  Curved belts map
    speed over radius to PhysX's degree-per-second angular convention.  PhysX rotates the surface
    field about the rigid-body origin, so ``-omega x pivot`` is included in the linear component
    to move the instantaneous center to :attr:`ConveyorBeltSpec.pivot_point`.

    Args:
        spec: Authored conveyor intent in the collision prim's local frame.
        velocity: Optional signed speed override [m/s]. Defaults to ``spec.velocity``.

    Returns:
        Local linear and angular PhysX surface velocities.

    Raises:
        ValueError: If the speed is not finite, the direction is zero, or a curved belt has no
            positive radius.
    """
    speed = spec.velocity if velocity is None else _finite_float("velocity", velocity)
    direction = np.asarray(spec.direction, dtype=np.float64)
    magnitude = float(np.linalg.norm(direction))
    if not math.isfinite(magnitude) or magnitude <= 1.0e-8:
        raise ValueError(f"Conveyor direction must be finite and non-zero, got {spec.direction!r}.")
    unit_direction = direction / magnitude

    if not spec.curved:
        linear = unit_direction * speed
        return PhysxSurfaceVelocityTwist(tuple(float(value) for value in linear), (0.0, 0.0, 0.0))

    if spec.radius is None or not math.isfinite(spec.radius) or spec.radius <= 0.0:
        raise ValueError("A curved PhysX conveyor requires a finite positive radius.")
    angular_rad = unit_direction * (speed / spec.radius)
    pivot = np.asarray(spec.pivot_point, dtype=np.float64)
    linear = -np.cross(angular_rad, pivot)
    angular_deg = np.degrees(angular_rad)
    return PhysxSurfaceVelocityTwist(
        tuple(float(value) for value in linear), tuple(float(value) for value in angular_deg)
    )


def apply_physx_surface_velocity_api(
    prim_or_path: Any,
    spec: ConveyorBeltSpec,
    *,
    velocity_scale: float = 0.0,
    stage: Any | None = None,
) -> None:
    """Apply and author a kinematic PhysX surface-velocity API on one belt prim.

    This function is intended for a task spawner, before PhysX parses the stage.  It lazily imports
    ``pxr.PhysxSchema`` and applies both the rigid-body and surface-velocity schemas.  The rigid body
    is made kinematic, surface velocities are authored in local space, and the initial command is
    scaled explicitly.  A zero default prevents pre-runtime motion during simulation warmup.

    Args:
        prim_or_path: USD prim object or exact prim path.
        spec: Conveyor intent used to author the local twist.
        velocity_scale: Finite multiplier for the initial signed surface speed.
        stage: Optional USD stage used when ``prim_or_path`` is a string. Defaults to the current
            Isaac Lab stage.

    Raises:
        RuntimeError: If the PhysX schema is unavailable or the target prim does not exist.
        ValueError: If ``velocity_scale`` is not finite.
    """
    scale = _finite_float("velocity_scale", velocity_scale)
    twist = compute_physx_surface_velocity_twist(spec, velocity=spec.velocity * scale)
    binding = _PhysxSchemaSurfaceWriter((prim_or_path,), stage=stage, apply_api=True)
    try:
        binding.write(0, enabled=spec.enabled, twist=twist)
    finally:
        binding.close()


def resolve_physx_conveyor_paths(
    num_envs: int,
    belt_specs: Sequence[ConveyorBeltSpec],
    env_path_format: str = "/World/envs/env_{}",
) -> tuple[str, ...]:
    """Resolve conveyor templates to exact environment-major prim paths.

    Args:
        num_envs: Number of replicated environments.
        belt_specs: Within-environment belt descriptions.
        env_path_format: Exact environment path format containing one ``{}`` field.

    Returns:
        Exact paths ordered by environment, then by ``belt_specs`` order.

    Raises:
        ValueError: If inputs cannot produce one unique path per environment and belt.
    """
    if not isinstance(num_envs, int) or isinstance(num_envs, bool) or num_envs <= 0:
        raise ValueError(f"Conveyor num_envs must be a positive integer, got {num_envs!r}.")
    specs = tuple(belt_specs)
    if not specs or not all(isinstance(spec, ConveyorBeltSpec) for spec in specs):
        raise ValueError("Conveyor belt_specs must contain at least one ConveyorBeltSpec.")
    if not isinstance(env_path_format, str) or env_path_format.count("{}") != 1:
        raise ValueError(f"Conveyor env_path_format must contain exactly one '{{}}', got {env_path_format!r}.")
    try:
        env_paths = tuple(env_path_format.format(env_id) for env_id in range(num_envs))
    except (IndexError, KeyError, ValueError) as exc:
        raise ValueError(f"Invalid conveyor env_path_format: {env_path_format!r}.") from exc
    if any(not path.startswith("/") or "{" in path or "}" in path for path in env_paths):
        raise ValueError(f"Conveyor env_path_format must produce exact absolute paths, got {env_path_format!r}.")

    templates = tuple(spec.prim_path for spec in specs)
    if len(set(templates)) != len(templates):
        raise ValueError("Conveyor belt spec paths must be unique within an environment.")
    if num_envs > 1 and any(_ENV_REGEX_NS not in path for path in templates):
        raise ValueError("Replicated PhysX conveyors require every belt prim_path to use {ENV_REGEX_NS}.")

    paths = tuple(template.replace(_ENV_REGEX_NS, env_path) for env_path in env_paths for template in templates)
    if len(set(paths)) != len(paths):
        raise ValueError("Resolved PhysX conveyor prim paths must be unique.")
    return paths


class PhysxSurfaceVelocityConveyor:
    """Host-side CPU reference facade for native PhysX surface velocity.

    The facade binds exact environment-major paths whose schemas were already authored by
    :func:`apply_physx_surface_velocity_api`.  Call :meth:`start` to register physics-rate updates,
    or call :meth:`update` manually.  Commands and enabled state survive full resets; a full reset
    clears encoders and restarts the one-second startup ramp. This facade authors USD attributes
    on the host and is not a GPU conveyor implementation.
    """

    def __init__(
        self,
        num_envs: int,
        belt_specs: Sequence[ConveyorBeltSpec],
        *,
        env_path_format: str = "/World/envs/env_{}",
        startup_duration_s: float = 1.0,
        stage: Any | None = None,
        writer: Any | None = None,
    ) -> None:
        """Bind native surface attributes and initialize host-side control state.

        Args:
            num_envs: Number of replicated environments.
            belt_specs: Within-environment belt descriptions.
            env_path_format: Exact replicated environment path format.
            startup_duration_s: Duration of the global surface-speed ramp [s].
            stage: Optional USD stage used by the default schema writer.
            writer: Optional writer implementing ``write(index, enabled=..., twist=...)`` and
                ``close()``. This import-light seam is primarily for focused tests.
        """
        duration = _finite_float("startup_duration_s", startup_duration_s)
        if duration <= 0.0:
            raise ValueError(f"Conveyor startup_duration_s must be positive, got {startup_duration_s!r}.")
        self._num_envs = num_envs
        self._belt_specs = tuple(belt_specs)
        self._surface_paths = resolve_physx_conveyor_paths(num_envs, self._belt_specs, env_path_format)
        self._belts_per_env = len(self._belt_specs)
        self._startup_duration_s = duration
        self._elapsed_time = 0.0
        self._velocity_scale = 0.0
        self._closed = False
        self._callback_handle: Any | None = None
        self._writer: _SurfaceVelocityWriter = (
            writer
            if writer is not None
            else _PhysxSchemaSurfaceWriter(self._surface_paths, stage=stage, apply_api=False)
        )

        self._command_velocity = np.tile(
            np.asarray([spec.velocity for spec in self._belt_specs], dtype=np.float32), self._num_envs
        )
        self._enabled = np.tile(np.asarray([spec.enabled for spec in self._belt_specs], dtype=np.bool_), self._num_envs)
        self._friction = np.tile(
            np.asarray([spec.friction_coefficient for spec in self._belt_specs], dtype=np.float32), self._num_envs
        )
        self._threshold = np.tile(
            np.asarray([spec.contact_threshold for spec in self._belt_specs], dtype=np.float32), self._num_envs
        )
        self._encoder_position = np.zeros(self.num_belts, dtype=np.float32)
        self._last_authored: list[tuple[bool, PhysxSurfaceVelocityTwist] | None] = [None] * self.num_belts
        self._flush(force=True)

    @property
    def specs(self) -> tuple[ConveyorBeltSpec, ...]:
        """Return authored descriptions in stable within-environment order."""
        return self._belt_specs

    @property
    def belts_per_env(self) -> int:
        """Return the number of authored belts per environment."""
        return self._belts_per_env

    @property
    def prim_paths(self) -> tuple[str, ...]:
        """Return exact PhysX prim paths in environment-major order."""
        return self._surface_paths

    @property
    def surface_paths(self) -> tuple[str, ...]:
        """Return an alias for :attr:`prim_paths`."""
        return self.prim_paths

    @property
    def num_belts(self) -> int:
        """Return the total number of bound conveyor surfaces."""
        return len(self._surface_paths)

    @property
    def count(self) -> int:
        """Return an alias for :attr:`num_belts`."""
        return self.num_belts

    @property
    def initialized(self) -> bool:
        """Return whether the facade still owns live schema bindings."""
        return not self._closed

    def start(self) -> None:
        """Register one lazy PhysX post-step callback; repeated calls are safe."""
        self._require_open()
        if self._callback_handle is not None:
            return
        from isaaclab_physx.physics import IsaacEvents, PhysxManager

        self._callback_handle = PhysxManager.register_callback(
            self.update,
            IsaacEvents.POST_PHYSICS_STEP,
            name="physx_conveyor_surface_velocity",
        )

    def update(self, dt: float) -> None:
        """Advance encoder state and the startup ramp by one physics step.

        Args:
            dt: Positive physics step duration [s].
        """
        self._require_open()
        step = _finite_float("physics dt", dt)
        if step <= 0.0:
            raise ValueError(f"Conveyor physics dt must be positive, got {dt!r}.")
        effective_velocity = self._command_velocity * self._enabled
        self._encoder_position += np.asarray(step * effective_velocity, dtype=np.float32)
        self._elapsed_time = min(self._startup_duration_s, self._elapsed_time + step)
        self._velocity_scale = self._elapsed_time / self._startup_duration_s
        self._flush()

    def set_velocities(self, velocities: Any, indices: Any = None) -> None:
        """Set signed speeds [m/s], preserving commands while belts are disabled."""
        selected = self._resolve_indices(indices)
        self._command_velocity[selected] = self._broadcast_finite(velocities, len(selected), "velocities")
        self._flush(selected)

    def get_velocities(self, indices: Any = None, clone: bool = True) -> np.ndarray:
        """Return effective speeds with disabled belts reported as zero."""
        return self._get_values(self._command_velocity * self._enabled, indices, clone)

    def get_commanded_velocities(self, indices: Any = None, clone: bool = True) -> np.ndarray:
        """Return commanded speeds before applying the enabled mask."""
        return self._get_values(self._command_velocity, indices, clone)

    def set_enabled(self, flags: Any, indices: Any = None) -> None:
        """Enable or disable belts without discarding their commands."""
        selected = self._resolve_indices(indices)
        values = _as_numpy(flags)
        if values.ndim == 0:
            values = np.full(len(selected), values.item())
        else:
            values = values.reshape(-1)
            if values.size == 1:
                values = np.full(len(selected), values.item())
        if values.size != len(selected) or not np.all(np.isin(values, (False, True, 0, 1))):
            raise ValueError(f"Conveyor enabled flags must contain {len(selected)} boolean values.")
        self._enabled[selected] = values.astype(np.bool_)
        self._flush(selected)

    def get_enabled(self, indices: Any = None, clone: bool = True) -> np.ndarray:
        """Return enabled flags as integer values."""
        return self._get_values(self._enabled.astype(np.int32), indices, clone)

    def set_friction_coefficients(self, coefficients: Any, indices: Any = None) -> None:
        """Reject unsupported runtime friction mutation explicitly."""
        del coefficients, indices
        raise NotImplementedError(
            "Native PhysX surface velocity uses authored collision materials; mutate material friction explicitly."
        )

    def get_friction_coefficients(self, indices: Any = None, clone: bool = True) -> np.ndarray:
        """Return authored friction metadata retained for control introspection."""
        return self._get_values(self._friction, indices, clone)

    def set_contact_processing_thresholds(self, thresholds: Any, indices: Any = None) -> None:
        """Reject unsupported normal-threshold mutation explicitly."""
        del thresholds, indices
        raise NotImplementedError(
            "PhysxSurfaceVelocityAPI has no contact-normal threshold; the native body-level field affects all contacts."
        )

    def get_contact_processing_thresholds(self, indices: Any = None, clone: bool = True) -> np.ndarray:
        """Return authored threshold metadata retained for control introspection."""
        return self._get_values(self._threshold, indices, clone)

    def get_encoder_positions(self, indices: Any = None, clone: bool = True) -> np.ndarray:
        """Return physics-rate integrated commanded belt travel [m]."""
        return self._get_values(self._encoder_position, indices, clone)

    def reset(self, env_ids: Any = None) -> None:
        """Clear selected encoders and restart the global ramp on a full reset.

        Velocity commands and enabled state are deliberately preserved, including across a full
        hard-reset path.  Partial resets do not disturb the global ramp used by other environments.

        Args:
            env_ids: Environment indices to reset, or ``None`` for all environments.
        """
        self._require_open()
        ids = self._resolve_env_ids(env_ids)
        rows = (ids[:, None] * self._belts_per_env + np.arange(self._belts_per_env)[None, :]).reshape(-1)
        self._encoder_position[rows] = 0.0
        if len(np.unique(ids)) == self._num_envs:
            self._elapsed_time = 0.0
            self._velocity_scale = 0.0
            self._flush(force=True)

    def close(self) -> None:
        """Deregister callbacks, disable authored motion, and release bindings safely."""
        if self._closed:
            return
        if self._callback_handle is not None:
            self._callback_handle.deregister()
            self._callback_handle = None
        zero = PhysxSurfaceVelocityTwist((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        for index in range(self.num_belts):
            self._writer.write(index, enabled=False, twist=zero)
        self._writer.close()
        self._closed = True

    def _flush(self, indices: Any = None, *, force: bool = False) -> None:
        """Author scaled effective commands for selected rows."""
        selected = self._resolve_indices(indices)
        for index in selected:
            enabled = bool(self._enabled[index])
            spec = self._belt_specs[index % self._belts_per_env]
            speed = float(self._command_velocity[index]) * self._velocity_scale if enabled else 0.0
            twist = compute_physx_surface_velocity_twist(spec, velocity=speed)
            state = (enabled, twist)
            if force or state != self._last_authored[index]:
                self._writer.write(int(index), enabled=enabled, twist=twist)
                self._last_authored[index] = state

    def _resolve_indices(self, indices: Any) -> np.ndarray:
        """Normalize and validate a belt row selection."""
        self._require_open()
        if indices is None:
            return np.arange(self.num_belts, dtype=np.int64)
        if isinstance(indices, slice):
            return np.arange(self.num_belts, dtype=np.int64)[indices]
        selected = _as_numpy(indices)
        if selected.dtype == np.bool_:
            if selected.ndim != 1 or selected.size != self.num_belts:
                raise IndexError(f"Boolean conveyor indices must have length {self.num_belts}.")
            return np.flatnonzero(selected).astype(np.int64)
        if not np.issubdtype(selected.dtype, np.integer):
            raise IndexError(f"Conveyor indices must be integers, got {indices!r}.")
        selected = selected.astype(np.int64, copy=False).reshape(-1)
        if np.any((selected < 0) | (selected >= self.num_belts)):
            raise IndexError(f"Conveyor surface indices are out of range: {selected.tolist()}.")
        return selected

    def _resolve_env_ids(self, env_ids: Any) -> np.ndarray:
        """Normalize and validate an environment selection."""
        if env_ids is None:
            return np.arange(self._num_envs, dtype=np.int64)
        ids = _as_numpy(env_ids)
        if not np.issubdtype(ids.dtype, np.integer):
            raise IndexError(f"Conveyor reset environment indices must be integers, got {env_ids!r}.")
        ids = ids.astype(np.int64, copy=False).reshape(-1)
        if np.any((ids < 0) | (ids >= self._num_envs)):
            raise IndexError(f"Conveyor reset environment indices are out of range: {ids.tolist()}.")
        return ids

    @staticmethod
    def _broadcast_finite(values: Any, count: int, name: str) -> np.ndarray:
        """Return a finite float32 vector with scalar broadcasting."""
        try:
            result = np.asarray(_as_numpy(values), dtype=np.float32)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"Conveyor {name} must contain finite numeric values.") from exc
        if result.ndim == 0:
            result = np.full(count, result.item(), dtype=np.float32)
        else:
            result = result.reshape(-1)
            if result.size == 1:
                result = np.full(count, result.item(), dtype=np.float32)
        if result.size != count or not np.all(np.isfinite(result)):
            raise ValueError(f"Conveyor {name} must contain one or {count} finite values.")
        return result

    def _get_values(self, source: np.ndarray, indices: Any, clone: bool) -> np.ndarray:
        """Return a complete buffer or a selected copy."""
        self._require_open()
        if indices is None:
            return source.copy() if clone else source
        return source[self._resolve_indices(indices)].copy()

    def _require_open(self) -> None:
        """Fail predictably after backend resources have been released."""
        if self._closed:
            raise RuntimeError("The PhysX conveyor surface facade is closed.")


class _SurfaceVelocityWriter(Protocol):
    """Minimal schema-writer seam used by the runtime facade."""

    def write(self, index: int, *, enabled: bool, twist: PhysxSurfaceVelocityTwist) -> None:
        """Author one bound surface state."""
        ...

    def close(self) -> None:
        """Release retained schema attributes."""
        ...


class _PhysxSchemaSurfaceWriter:
    """Cached USD attribute writer with all schema imports kept lazy."""

    def __init__(self, prims_or_paths: Sequence[Any], *, stage: Any | None, apply_api: bool) -> None:
        """Resolve prims, optionally apply schemas, and cache surface attributes."""
        try:
            from pxr import Gf, PhysxSchema, UsdPhysics
        except ImportError as exc:
            raise RuntimeError(
                "PhysxSurfaceVelocityAPI is unavailable. Use this backend inside an Isaac Sim process with PhysX "
                "schemas."
            ) from exc

        if stage is None and any(isinstance(value, str) for value in prims_or_paths):
            from isaaclab.sim.utils.stage import get_current_stage

            stage = get_current_stage()
        self._gf = Gf
        self._attributes: list[tuple[Any, Any, Any]] = []
        for value in prims_or_paths:
            prim = stage.GetPrimAtPath(value) if isinstance(value, str) else value
            if prim is None or not prim.IsValid():
                raise RuntimeError(f"Cannot bind PhysX conveyor surface: prim {value!r} does not exist.")

            rigid_body = UsdPhysics.RigidBodyAPI(prim)
            if apply_api and not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                rigid_body = UsdPhysics.RigidBodyAPI.Apply(prim)
            elif not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                raise RuntimeError(f"PhysX conveyor prim {prim.GetPath()} has no authored RigidBodyAPI.")
            if apply_api:
                rigid_body.CreateRigidBodyEnabledAttr().Set(True)
                rigid_body.CreateKinematicEnabledAttr().Set(True)
            elif rigid_body.GetKinematicEnabledAttr().Get() is not True:
                raise RuntimeError(f"PhysX conveyor prim {prim.GetPath()} must be an authored kinematic rigid body.")

            if prim.HasAPI(PhysxSchema.PhysxSurfaceVelocityAPI):
                surface_api = PhysxSchema.PhysxSurfaceVelocityAPI(prim)
            elif apply_api:
                surface_api = PhysxSchema.PhysxSurfaceVelocityAPI.Apply(prim)
            else:
                raise RuntimeError(
                    f"PhysX conveyor prim {prim.GetPath()} has no authored PhysxSurfaceVelocityAPI; "
                    "call apply_physx_surface_velocity_api from its spawner before simulation starts."
                )
            surface_api.CreateSurfaceVelocityLocalSpaceAttr().Set(True)
            self._attributes.append(
                (
                    surface_api.CreateSurfaceVelocityEnabledAttr(),
                    surface_api.CreateSurfaceVelocityAttr(),
                    surface_api.CreateSurfaceAngularVelocityAttr(),
                )
            )

    def write(self, index: int, *, enabled: bool, twist: PhysxSurfaceVelocityTwist) -> None:
        """Author one cached local-space surface twist."""
        enabled_attr, linear_attr, angular_attr = self._attributes[index]
        # PhysX caches the contact-modification flag when a surface-velocity
        # shape is parsed.  Cycle it around command changes, matching the
        # official Isaac Conveyor node, so live updates are picked up without
        # leaving a stale contact modifier attached to the shape.
        enabled_attr.Set(False)
        linear_attr.Set(self._gf.Vec3f(*twist.linear_velocity))
        angular_attr.Set(self._gf.Vec3f(*twist.angular_velocity_deg))
        enabled_attr.Set(enabled)

    def close(self) -> None:
        """Release cached schema attribute handles."""
        self._attributes.clear()


def _finite_float(name: str, value: Any) -> float:
    """Coerce one finite scalar with a stable validation error."""
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"Conveyor {name} must be finite, got {value!r}.") from exc
    if not math.isfinite(result):
        raise ValueError(f"Conveyor {name} must be finite, got {value!r}.")
    return result


def _as_numpy(values: Any) -> np.ndarray:
    """Move supported tensor-like values to a host NumPy array without importing their libraries."""
    if isinstance(values, np.ndarray):
        return values
    if hasattr(values, "detach"):
        values = values.detach()
    if hasattr(values, "cpu"):
        values = values.cpu()
    if hasattr(values, "numpy"):
        return np.asarray(values.numpy())
    return np.asarray(values)
