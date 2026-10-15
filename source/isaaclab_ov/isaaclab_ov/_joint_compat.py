# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Preserve supported legacy joint opinions in the private OVPhysX 0.7 USD export."""

from __future__ import annotations

import math
import re

from pxr import Sdf


def convert_legacy_joint_properties(
    layer: Sdf.Layer, friction_replacements: dict[str, tuple[float, float, float]] | None = None
) -> None:
    """Translate authored joint properties, including flattened instance prototypes, in place.

    Call only on the temporary export layer. Both namespaces store angular velocity
    in degrees per second and armature in mass times length squared. Existing Newton
    opinions and per-axis PhysX opinions retain precedence. Friction coefficients have
    no equivalent in the new effort-based schema and must be migrated explicitly.
    """
    replacements = []
    for pattern, values in (friction_replacements or {}).items():
        if len(values) != 3 or not all(math.isfinite(value) and value >= 0 for value in values):
            raise ValueError(f"Invalid joint friction replacement for {pattern}: expected three finite nonnegative values.")
        if values[0] < values[1]:
            raise ValueError(f"Invalid joint friction replacement for {pattern}: static effort must be >= dynamic effort.")
        replacements.append((re.compile(pattern), values))
    paths = []
    layer.Traverse(Sdf.Path.absoluteRootPath, paths.append)
    for path in paths:
        prim = layer.GetPrimAtPath(path)
        if prim is None:
            continue
        friction = prim.attributes.get("physxJoint:jointFriction")
        if friction is not None and (friction.default not in (None, 0.0, Sdf.ValueBlock()) or layer.ListTimeSamplesForPath(friction.path)):
            matches = [values for pattern, values in replacements if pattern.fullmatch(prim.name)]
            if len(matches) > 1:
                raise ValueError(f"Multiple joint friction replacements match {path}.")
            axis = {"PhysicsRevoluteJoint": "angular", "PhysicsPrismaticJoint": "linear"}.get(prim.typeName)
            if not matches or axis is None or layer.ListTimeSamplesForPath(friction.path):
                raise ValueError(
                    f"OVPhysX 0.7 cannot preserve legacy joint friction on {path}. "
                    "Remove physxJoint:jointFriction and retune NewtonJointAPI newton:friction/newton:damping "
                    "or configure legacy_joint_friction_replacements with calibrated efforts; "
                    "the old coefficient has no exact conversion."
                )
            static_name = f"physxJointAxis:{axis}:staticFrictionEffort"
            dynamic_name = f"physxJointAxis:{axis}:dynamicFrictionEffort"
            # Keep existing modern opinions, including explicit zeros. Migration supplies only
            # missing properties; the native parser owns Newton/per-axis precedence.
            modern_names = ("newton:friction", static_name, dynamic_name)
            schemas = list(prim.GetInfo("apiSchemas").GetAppliedItems()) if prim.HasInfo("apiSchemas") else []
            schema = f"PhysxJointAxisAPI:{axis}"
            if schema not in schemas:
                prim.SetInfo("apiSchemas", Sdf.TokenListOp.CreateExplicit([*schemas, schema]))
            if not any(name in prim.attributes for name in modern_names):
                for name, value in zip((static_name, dynamic_name), matches[0][:2], strict=True):
                    Sdf.AttributeSpec(prim, name, Sdf.ValueTypeNames.Float).default = value
            viscous_name = f"physxJointAxis:{axis}:viscousFrictionCoefficient"
            if "newton:damping" not in prim.attributes and viscous_name not in prim.attributes:
                # Angular USD damping is effort*s/degree; runtime coefficients use radians.
                value = matches[0][2] * (math.pi / 180.0 if axis == "angular" else 1.0)
                Sdf.AttributeSpec(prim, viscous_name, Sdf.ValueTypeNames.Float).default = value
            prim.RemoveProperty(friction)
        for old_name, new_name, default in (
            ("physxJoint:maxJointVelocity", "newton:velocityLimit", float("inf")),
            ("physxJoint:armature", "newton:armature", 0.0),
        ):
            old = prim.attributes.get(old_name)
            if old is None:
                continue
            modern = prim.attributes.get(new_name)
            if modern is not None and (modern.HasDefaultValue() or layer.ListTimeSamplesForPath(modern.path)):
                continue
            if layer.ListTimeSamplesForPath(old.path):
                raise ValueError(f"OVPhysX 0.7 cannot preserve time-sampled {old_name} on {path}; migrate to {new_name}.")
            if not old.HasDefaultValue() or old.default == Sdf.ValueBlock():
                continue
            if prim.typeName not in ("PhysicsRevoluteJoint", "PhysicsPrismaticJoint", "PhysicsSphericalJoint", "PhysicsJoint"):
                if old.default != default:
                    raise ValueError(f"OVPhysX 0.7 cannot preserve {old_name} on {prim.typeName} at {path}.")
                continue
            schemas = list(prim.GetInfo("apiSchemas").GetAppliedItems()) if prim.HasInfo("apiSchemas") else []
            if "NewtonJointAPI" not in schemas:
                prim.SetInfo("apiSchemas", Sdf.TokenListOp.CreateExplicit([*schemas, "NewtonJointAPI"]))
            attribute = modern or Sdf.AttributeSpec(prim, new_name, old.typeName)
            attribute.default = old.default
