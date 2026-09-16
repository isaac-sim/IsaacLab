# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Physical source-unit expressions and the explicitly maintained USD target rules."""

import math
import re

import numpy as np

from pxr import UsdGeom, UsdPhysics


class UnitConverter:
    """Convert compatible products of length, mass, time and angle units."""

    @staticmethod
    def resolve(expression: str, *, length: float = 1.0, mass: float = 1.0) -> tuple[tuple[int, ...], float]:
        """Resolve a unit expression into dimensions and its SI scale."""
        units = {
            "1": ((0, 0, 0, 0), 1.0),
            "m": ((1, 0, 0, 0), 1.0),
            "kg": ((0, 1, 0, 0), 1.0),
            "s": ((0, 0, 1, 0), 1.0),
            "rad": ((0, 0, 0, 1), 1.0),
            "deg": ((0, 0, 0, 1), math.pi / 180),
            "N": ((1, 1, -2, 0), 1.0),
            "length": ((1, 0, 0, 0), length),
            "mass": ((0, 1, 0, 0), mass),
        }
        dimensions, scale, sign = np.zeros(4, dtype=int), 1.0, 1
        for token in re.split(r"([*/])", expression):
            if token in {"*", "/"}:
                sign = -1 if token == "/" else 1
                continue
            name, _, exponent = token.partition("^")
            if name not in units:
                raise ValueError(f"Unknown physical unit {name!r}.")
            power = sign * (int(exponent) if exponent else 1)
            dimension, factor = units[name]
            dimensions += np.array(dimension) * power
            scale *= factor**power
        return tuple(dimensions), scale

    @classmethod
    def convert(cls, value, source: str, target: str, *, length: float = 1.0, mass: float = 1.0):
        """Scale a value without interpreting its field name or physical owner."""
        source_dimension, source_scale = cls.resolve(source, length=length, mass=mass)
        target_dimension, target_scale = cls.resolve(target, length=length, mass=mass)
        if source_dimension != target_dimension:
            raise ValueError(f"Incompatible physical units: {source} and {target}.")
        # Preserve integer flags and exact values when no conversion is necessary.
        return np.asarray(value) if source_scale == target_scale else np.asarray(value) * (source_scale / target_scale)


class UsdPhysicsUnits:
    """USD units maintained from schema semantics; USD has no complete unit metadata API."""

    @staticmethod
    def for_attribute(prim, declaration, axis: str | None) -> str:
        """Return the target unit only for a known attribute on a compatible prim."""
        name = declaration.attribute.format(axis=axis)
        if prim.HasAPI(UsdPhysics.RigidBodyAPI) or prim.HasAPI(UsdPhysics.MassAPI):
            body = {
                "physics:mass": "mass",
                "physics:density": "mass/length^3",
                "physics:centerOfMass": "length",
                "physics:diagonalInertia": "mass*length^2",
                "physics:principalAxes": "1",
                "physxRigidBody:disableGravity": "1",
            }
            if name in body:
                return body[name]
        if prim.IsA(UsdPhysics.Joint):
            angular = axis in {"angular", "rotX", "rotY", "rotZ"}
            if not angular and axis not in {"linear", "transX", "transY", "transZ"}:
                raise ValueError(f"Missing joint axis for {prim.GetPath()}.{name}.")
            position = "deg" if angular else "length"
            effort = "mass*length^2/s^2" if angular else "mass*length/s^2"
            stiffness = effort + "/deg" if angular else "mass/s^2"
            damping = effort + "*s/deg" if angular else "mass/s"
            suffix = name.rsplit(":", 1)[-1]
            if name in {"physics:lowerLimit", "physics:upperLimit"}:
                expected = UsdPhysics.RevoluteJoint if angular else UsdPhysics.PrismaticJoint
                if not prim.IsA(expected):
                    raise ValueError(f"Scalar limit does not describe {prim.GetTypeName()}.")
                return position
            if declaration.schema and declaration.schema.startswith("PhysicsLimitAPI:") and suffix in {"low", "high"}:
                return position
            if declaration.schema and declaration.schema.startswith("PhysicsDriveAPI:"):
                return {"stiffness": stiffness, "damping": damping, "maxForce": effort}[suffix]
            # Extension aliases share these documented physical semantics.
            if name.startswith(("physxJoint:", "physxJointAxis:", "newton:")):
                return {
                    "maxJointVelocity": position + "/s",
                    "velocityLimit": position + "/s",
                    "armature": "mass*length^2" if angular else "mass",
                    "staticFrictionEffort": effort,
                    "dynamicFrictionEffort": effort,
                    "friction": effort,
                    "frictionCoeff": "1",
                    "viscousFrictionCoefficient": damping,
                    "limitStiffness": stiffness,
                    "limitDamping": damping,
                }[suffix]
        if prim.IsA(UsdGeom.BasisCurves) and name.startswith("newton:export:shape_"):
            name = name.removeprefix("newton:export:shape_").removeprefix("material_")
            return {
                "margin": "length",
                "gap": "length",
                "mu": "1",
                "restitution": "1",
                "ke": "mass/s^2",
                "kd": "mass/s",
                "kf": "mass/s",
                "ka": "length",
                "mu_torsional": "length",
                "mu_rolling": "length",
            }[name]
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            if name in {
                "physxCollision:contactOffset",
                "physxCollision:restOffset",
                "newton:contactMargin",
                "newton:contactGap",
            }:
                return "length"
        # Material properties may be supplied before applying their API.
        if prim.GetTypeName() == "Material":
            return {
                "physics:staticFriction": "1",
                "physics:dynamicFriction": "1",
                "physics:restitution": "1",
                "newton:torsionalFriction": "length",
                "newton:rollingFriction": "length",
                "newton:contactStiffness": "mass/s^2",
                "newton:contactDamping": "mass/s",
                "newton:contactFrictionGain": "mass/s",
                "newton:contactAdhesion": "length",
            }[name]
        raise NotImplementedError(f"No USD unit rule for {prim.GetPath()}.{name}.")

    @classmethod
    def convert(cls, prim, declaration, value, source: str, axis: str | None):
        """Convert declared physical values into this stage's units."""
        return UnitConverter.convert(
            value,
            source,
            cls.for_attribute(prim, declaration, axis),
            length=UsdGeom.GetStageMetersPerUnit(prim.GetStage()),
            mass=UsdPhysics.GetStageKilogramsPerUnit(prim.GetStage()),
        )
