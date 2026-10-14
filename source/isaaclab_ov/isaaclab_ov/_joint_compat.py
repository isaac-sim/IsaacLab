# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Preserve supported legacy joint opinions in the private OVPhysX 0.7 USD export."""

from __future__ import annotations

from pxr import Sdf


def convert_legacy_joint_properties(layer: Sdf.Layer) -> None:
    """Translate authored joint properties, including flattened instance prototypes, in place.

    Call only on the temporary export layer. Both namespaces store angular velocity
    in degrees per second and armature in mass times length squared. Existing Newton
    opinions and per-axis PhysX opinions retain precedence. Friction coefficients have
    no equivalent in the new effort-based schema and must be migrated explicitly.
    """
    paths = []
    layer.Traverse(Sdf.Path.absoluteRootPath, paths.append)
    for path in paths:
        prim = layer.GetPrimAtPath(path)
        if prim is None:
            continue
        friction = prim.attributes.get("physxJoint:jointFriction")
        if friction is not None and (friction.default not in (None, 0.0, Sdf.ValueBlock()) or layer.ListTimeSamplesForPath(friction.path)):
            raise ValueError(
                f"OVPhysX 0.7 cannot preserve legacy joint friction on {path}. "
                "Remove physxJoint:jointFriction and retune NewtonJointAPI newton:friction/newton:damping "
                "or PhysxJointAxisAPI friction efforts; the old coefficient has no exact conversion."
            )
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
