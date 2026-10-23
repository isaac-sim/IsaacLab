# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Native CPU PhysX conveyor surfaces with a resettable startup ramp."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .surface_velocity import SurfaceVelocitySpec, resolve_surface_velocity_paths

if TYPE_CHECKING:
    import torch

    from pxr import Usd


@dataclass(frozen=True, slots=True)
class PhysxSurfaceVelocityTwist:
    """Local linear velocity [m/s] and angular velocity [deg/s]."""

    linear_velocity: tuple[float, float, float]
    angular_velocity_deg: tuple[float, float, float]


def compute_surface_velocity_twist(spec: SurfaceVelocitySpec) -> PhysxSurfaceVelocityTwist:
    """Convert a surface speed to PhysX units, rotating curves about their local pivot."""
    direction = np.asarray(spec.direction, dtype=np.float64)
    direction /= np.linalg.norm(direction)
    if not spec.curved:
        return PhysxSurfaceVelocityTwist(tuple(direction * spec.velocity), (0.0, 0.0, 0.0))
    angular_rad = direction * (spec.velocity / spec.radius)
    return PhysxSurfaceVelocityTwist(tuple(-np.cross(angular_rad, spec.pivot_point)), tuple(np.degrees(angular_rad)))


def apply_surface_velocity_api(prim: Usd.Prim) -> None:
    """Author a kinematic local-space surface at zero speed before PhysX parses the stage."""
    from pxr import Gf

    zero = Gf.Vec3f(0.0)
    _write_surface(_surface_attributes(prim, apply_api=True), zero, zero, 0.0)


class SurfaceVelocity:
    """Bind authored CPU PhysX surfaces and ramp their fixed speeds after full resets."""

    def __init__(
        self,
        num_envs: int,
        surface_specs: Sequence[SurfaceVelocitySpec],
        *,
        stage: Usd.Stage,
        env_path_format: str = "/World/envs/env_{}",
        startup_duration_s: float = 1.0,
    ) -> None:
        """Cache native attributes after scene parsing; schema imports stay lazy."""
        from pxr import Gf

        specs = tuple(surface_specs)
        paths = resolve_surface_velocity_paths(num_envs, specs, env_path_format)
        if not math.isfinite(startup_duration_s) or startup_duration_s <= 0.0:
            raise ValueError("Conveyor startup_duration_s must be finite and positive.")
        self._num_envs = num_envs
        self._startup_duration_s = startup_duration_s
        self._elapsed_time = 0.0
        self._callback_handle = None
        self._surfaces = []
        for path, spec in zip(paths, specs * num_envs, strict=True):
            attributes = _surface_attributes(stage.GetPrimAtPath(path), apply_api=False)
            twist = compute_surface_velocity_twist(spec)
            self._surfaces.append((attributes, Gf.Vec3f(*twist.linear_velocity), Gf.Vec3f(*twist.angular_velocity_deg)))
        self._write(0.0)

    def start(self) -> None:
        """Register the physics-rate ramp callback once."""
        if not self._surfaces:
            raise RuntimeError("The PhysX conveyor is closed.")
        if self._callback_handle is None:
            from isaaclab_physx.physics.physx_manager import IsaacEvents, PhysxManager

            self._callback_handle = PhysxManager.register_callback(
                self._update, IsaacEvents.POST_PHYSICS_STEP, name="physx_surface_velocity"
            )

    def _update(self, dt: float) -> None:
        """Stop authoring USD once the startup ramp has completed."""
        if self._elapsed_time < self._startup_duration_s:
            self._elapsed_time = min(self._startup_duration_s, self._elapsed_time + dt)
            self._write(self._elapsed_time / self._startup_duration_s)

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        """Restart the shared ramp for a full reset; partial resets preserve other belts."""
        if env_ids is None or len(env_ids) == self._num_envs:
            self._elapsed_time = 0.0
            self._write(0.0)

    def close(self) -> None:
        """Detach the callback and disable surface motion; repeated closes are harmless."""
        if self._callback_handle is not None:
            self._callback_handle.deregister()
            self._callback_handle = None
        self._write(0.0, enabled=False)
        self._surfaces.clear()

    def _write(self, scale: float, *, enabled: bool = True) -> None:
        for attributes, linear, angular in self._surfaces:
            _write_surface(attributes, linear, angular, scale, enabled=enabled)


def _surface_attributes(prim: Usd.Prim, *, apply_api: bool) -> tuple[Usd.Attribute, ...]:
    """Author or validate native schemas, then bind the three velocity attributes."""
    from pxr import PhysxSchema, UsdPhysics

    if not prim:
        raise RuntimeError("Cannot bind a missing PhysX conveyor prim.")
    if apply_api:
        rigid_body = UsdPhysics.RigidBodyAPI.Apply(prim)
        rigid_body.CreateRigidBodyEnabledAttr().Set(True)
        rigid_body.CreateKinematicEnabledAttr().Set(True)
        surface = PhysxSchema.PhysxSurfaceVelocityAPI.Apply(prim)
        surface.CreateSurfaceVelocityLocalSpaceAttr().Set(True)
    else:
        if (
            not prim.HasAPI(UsdPhysics.RigidBodyAPI)
            or not UsdPhysics.RigidBodyAPI(prim).GetKinematicEnabledAttr().Get()
        ):
            raise RuntimeError(f"PhysX conveyor {prim.GetPath()} requires an authored kinematic rigid body.")
        if not prim.HasAPI(PhysxSchema.PhysxSurfaceVelocityAPI):
            raise RuntimeError(f"PhysX conveyor {prim.GetPath()} requires authored PhysxSurfaceVelocityAPI.")
        surface = PhysxSchema.PhysxSurfaceVelocityAPI(prim)
    return (
        surface.CreateSurfaceVelocityEnabledAttr(),
        surface.CreateSurfaceVelocityAttr(),
        surface.CreateSurfaceAngularVelocityAttr(),
    )


def _write_surface(attributes, linear, angular, scale: float, *, enabled: bool = True) -> None:
    """Cycle contact modification around velocity changes, matching Isaac's Conveyor node."""
    enabled_attr, linear_attr, angular_attr = attributes
    enabled_attr.Set(False)
    linear_attr.Set(linear * scale)
    angular_attr.Set(angular * scale)
    enabled_attr.Set(enabled)
