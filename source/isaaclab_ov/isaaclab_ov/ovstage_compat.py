# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""OVStage population and clone compatibility without loading the native runtime."""

from functools import cache
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING

from packaging.version import Version

if TYPE_CHECKING:
    import ovstage

try:
    OVSTAGE_VERSION = Version(version("ovstage"))
except PackageNotFoundError:
    OVSTAGE_VERSION = None

# The 0.4 release line needs CPU transform propagation before cloning, including development builds.
OVSTAGE_REQUIRES_CPU_HIERARCHY_BEFORE_CLONE = OVSTAGE_VERSION is not None and OVSTAGE_VERSION.release[:2] == (0, 4)


def populate_stage(
    stage: "ovstage.Stage", usda: str, *, ordinal: int, domains: "ovstage.PopulationDomain", physics: bool = False
) -> None:
    """Populate rendering and optional physics data across the OVStage 0.4 API transition.

    Older builds expose a physics domain bit; newer 0.4 builds require selectors.
    Detect the capability because both APIs shipped under the same release version.

    Args:
        stage: Native stage to populate.
        usda: Exported USD scene.
        ordinal: Population write ordinal, sealed by the caller after cloning.
        domains: Native population domains, normally rendering.
        physics: Whether an OVPhysX consumer also needs physics data.
    """
    import ovstage

    if physics:
        physics_domain = getattr(ovstage.PopulationDomain, "PHYSICS", None)
        if physics_domain is None:
            ovstage.population.open_usd_from_string_with_desc(
                stage, usda, ordinal=ordinal, time=ovstage.population.DEFAULT_TIME,
                descs=(ovstage.population.Desc(domains=domains), _physics_population_desc()),
            )  # fmt: skip
            return
        domains |= physics_domain
    ovstage.population.open_usd_from_string(stage, usda, ordinal=ordinal, domains=domains)


@cache
def _physics_population_desc() -> "ovstage.population.Desc":
    """Preserve physics properties and the USD producer defaults documented by OVPhysX.

    Copying schema fallbacks alone loses unit scaling and resolved bindings from the legacy physics domain.
    """
    import ovstage
    from ovstage.population import (
        DerivedProperty,
        DeriveWhen,
        Desc,
        Expression,
        PrimPredicate,
        PropertyPredicate,
        Selector,
    )

    scene = PrimPredicate.is_a_type("PhysicsScene")
    body = PrimPredicate.has_applied_schema("PhysicsRigidBodyAPI")
    collider = PrimPredicate.has_applied_schema("PhysicsCollisionAPI")
    meters_per_unit = Expression.stage_metadata("metersPerUnit")
    speed = Expression.divide(Expression.number(10.0), meters_per_unit)
    speed_squared = Expression.multiply(speed, speed)
    float_dtype = ovstage.DLDataType(code=ovstage.DLDataTypeCode.kDLFloat, bits=32, lanes=1)
    float_max = Expression.number(3.4028234663852886e38)
    defaults = (
        ("physics:gravityMagnitude", scene, Expression.divide(Expression.number(9.81), meters_per_unit)),
        ("physxScene:frictionOffsetThreshold", scene, Expression.divide(Expression.number(0.04), meters_per_unit)),
        ("physxScene:frictionCorrelationDistance", scene, Expression.divide(Expression.number(0.025), meters_per_unit)),
        ("physxRigidBody:sleepThreshold", body, Expression.multiply(Expression.number(5e-5), speed_squared)),
        ("physxRigidBody:stabilizationThreshold", body, Expression.multiply(Expression.number(1e-5), speed_squared)),
        ("physxRigidBody:maxDepenetrationVelocity", body, Expression.divide(Expression.number(3.0), meters_per_unit)),
        ("physxRigidBody:maxLinearVelocity", body, Expression.number(1e16)),
        ("physxRigidBody:maxContactImpulse", body, float_max),
        (
            "physxCollision:contactOffset",
            PrimPredicate.and_(collider, PrimPredicate.is_a_type("Plane")),
            Expression.divide(Expression.number(0.02), meters_per_unit),
        ),
        (
            "physxConvexGeometry:margin",
            PrimPredicate.and_(collider, PrimPredicate.is_a_type("Cylinder", "Cone")),
            Expression.number(0.0),
        ),
    )
    derived = [
        DerivedProperty(name, value, prim_predicate=predicate, when=DeriveWhen.WHEN_UNAUTHORED, dtype=float_dtype)
        for name, predicate, value in defaults
    ]
    for family, name in (
        ("PhysxJointAxisAPI", "physxJointAxis:__INSTANCE_NAME__:maxJointVelocity"),
        ("PhysxDrivePerformanceEnvelopeAPI", "physxDrivePerformanceEnvelope:__INSTANCE_NAME__:maxActuatorVelocity"),
    ):
        derived.append(
            DerivedProperty(
                name, float_max, when=DeriveWhen.WHEN_UNAUTHORED, dtype=float_dtype, multi_apply_schema_family=family
            )
        )
    derived.append(
        DerivedProperty(
            "physics:gravityDirection",
            Expression.multiply(Expression.number(-1.0), Expression.up_axis()),
            prim_predicate=scene,
            when=DeriveWhen.WHEN_UNAUTHORED,
            dtype=ovstage.DLDataType(code=ovstage.DLDataTypeCode.kDLFloat, bits=32, lanes=3),
            semantic=ovstage.AttributeSemantic.VECTOR,
        )
    )
    derived.append(
        DerivedProperty(
            "material:binding:physics",
            Expression.resolved_bound_material(
                "physics",
                PrimPredicate.has_applied_schema(
                    "PhysicsMaterialAPI",
                    "PhysxPBDMaterialAPI",
                    "OmniPhysicsBaseMaterialAPI",
                    "PhysxDeformableMaterialAPI",
                ),
            ),
            prim_predicate=PrimPredicate.or_(
                collider,
                body,
                PrimPredicate.has_applied_schema("OmniPhysicsBodyAPI", "OmniPhysicsDeformableBodyAPI"),
                PrimPredicate.is_a_type("GeomSubset"),
            ),
            dtype=ovstage.DLDataType(code=ovstage.DLDataTypeCode.kDLUInt, bits=64, lanes=1),
            is_array=True,
            semantic=ovstage.AttributeSemantic.RELATIONSHIP_PATH_ID,
        )
    )
    return Desc(
        selectors=(
            Selector(
                prim_predicate=PrimPredicate.all(),
                property_predicate=PropertyPredicate.all(),
                prim_metadata_paths=("customData:physics:localSpaceVelocities", "inactiveIds"),
                property_metadata_paths=("bindMaterialAs",),
                derived_properties=derived,
            ),
        ),
        stage_metadata_paths=("metersPerUnit", "kilogramsPerUnit", "upAxis"),
    )
