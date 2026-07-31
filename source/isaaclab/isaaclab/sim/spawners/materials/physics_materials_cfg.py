# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import MISSING
from typing import ClassVar

from isaaclab.sim.schemas.schemas_cfg import SchemaFragment
from isaaclab.utils.configclass import configclass

# Names that moved out of this submodule into ``isaaclab_physx.sim.spawners.materials.physics_materials_cfg``.
# Resolved lazily so callers using ``from isaaclab.sim.spawners.materials.physics_materials_cfg
# import RigidBodyMaterialCfg`` continue to work without importing ``isaaclab_physx`` at module
# load time.
_PHYSX_FORWARDS = frozenset(
    {
        "DeformableBodyMaterialCfg",
        "RigidBodyMaterialCfg",
        "SurfaceDeformableBodyMaterialCfg",
        "PhysxRigidBodyMaterialCfg",
        "PhysxDeformableBodyMaterialCfg",
        "PhysxSurfaceDeformableBodyMaterialCfg",
    }
)


def __getattr__(name):
    if name in _PHYSX_FORWARDS:
        try:
            from isaaclab_physx.sim.spawners.materials import physics_materials_cfg as _physx_mat_cfg
        except ImportError as e:
            raise ImportError(
                f"'isaaclab.sim.spawners.materials.physics_materials_cfg.{name}' has moved to"
                " 'isaaclab_physx.sim.spawners.materials.physics_materials_cfg'. Install the"
                " isaaclab_physx extension or update your import. This forwarding shim is scheduled"
                " for removal in 4.0."
            ) from e
        return getattr(_physx_mat_cfg, name)
    raise AttributeError(f"module 'isaaclab.sim.spawners.materials.physics_materials_cfg' has no attribute {name!r}")


@configclass
class PhysicsMaterialCfg:
    """Configuration parameters for creating a physics material.

    Physics materials are USD schemas applied to a material prim to define the physical properties
    related to the material. For example, the friction coefficient, restitution coefficient, etc.
    Subclasses author the schema of a specific backend, such as the standard ``UsdPhysics``
    attributes, the AOUSD deformable schemas, or the PhysX extensions.
    """

    func: Callable = MISSING
    """Function to use for creating the material."""


@configclass
class CableMaterialCfg(PhysicsMaterialCfg):
    """Physics material parameters for deformable curves."""

    _usd_namespace: ClassVar[str | None] = "physics"
    _usd_applied_schema: ClassVar[str | None] = "PhysicsCurvesDeformableMaterialAPI"

    func: Callable | str = "{DIR}.physics_materials:spawn_deformable_body_material"

    thickness: float = 0.001
    """The finite, positive full cable thickness (diameter) [m]. Defaults to 0.001 m."""

    density: float = 1000.0
    """The finite, positive cable density [kg/m^3]. Defaults to 1000 kg/m^3."""

    stretch_stiffness: float = 1.0e9
    """The finite, nonnegative cable stretch elastic modulus [Pa]. Defaults to 1e9 Pa."""

    bend_stiffness: float = 1.0e6
    """The finite, nonnegative cable bend elastic modulus [Pa]. Defaults to 1e6 Pa."""

    shear_stiffness: float | None = None
    """The finite, nonnegative cable shear elastic modulus [Pa].

    Defaults to None, in which case it is not authored and the solver falls back to
    :attr:`stretch_stiffness`.
    """

    twist_stiffness: float | None = None
    """The finite, nonnegative cable twist elastic modulus [Pa].

    Defaults to None, in which case it is not authored and the solver falls back to
    :attr:`bend_stiffness`.
    """

    def validate_config(self) -> None:
        """Validate cable material values."""
        for field in ("thickness", "density"):
            value = getattr(self, field)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"CableMaterialCfg {field} must be finite and greater than zero.")
        for field in ("stretch_stiffness", "bend_stiffness", "shear_stiffness", "twist_stiffness"):
            value = getattr(self, field)
            if value is None:
                continue
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"CableMaterialCfg {field} must be finite and nonnegative.")


@configclass
class RigidBodyMaterialBaseCfg(PhysicsMaterialCfg):
    """Solver-common physics-material parameters for rigid bodies.

    Contains the friction, restitution, and density fields from the `UsdPhysics.MaterialAPI`_ that
    are common across all simulation backends. For properties in the ``physxMaterial`` namespace
    (compliant-contact spring and combine modes), use
    :class:`~isaaclab_physx.sim.spawners.materials.PhysxRigidBodyMaterialCfg`.

    See :meth:`spawn_rigid_body_material` for more information.

    .. _UsdPhysics.MaterialAPI: https://openusd.org/dev/api/class_usd_physics_material_a_p_i.html
    """

    # -- Class metadata (not dataclass fields) --
    # ``static_friction`` / ``dynamic_friction`` / ``restitution`` write to ``physics:*``
    # (UsdPhysics standard attributes). The helper's per-declaring-class routing keeps
    # them under the base namespace even when the cfg is a PhysX subclass instance.
    _usd_namespace: ClassVar[str | None] = "physics"
    _usd_applied_schema: ClassVar[str | None] = None
    _usd_field_exceptions: ClassVar[dict] = {}

    func: Callable | str = "{DIR}.physics_materials:spawn_rigid_body_material"

    static_friction: float = 0.5
    """The static friction coefficient. Defaults to 0.5."""

    dynamic_friction: float = 0.5
    """The dynamic friction coefficient. Defaults to 0.5."""

    restitution: float = 0.0
    """The restitution coefficient. Defaults to 0.0."""

    density: float | None = None
    """The material density [kg/m^3]. Defaults to None, in which case it is not authored.

    Writes ``physics:density``. It is a fallback for collision shapes bound to this material;
    explicitly authored rigid-body mass or density takes precedence.
    """


@configclass
class RigidBodyMaterialFragment(SchemaFragment):
    """Marker base for rigid-body physics-material fragments; types the ``physics_material`` slot.

    A rigid-body physics material is a single ``UsdShade.Material`` prim that carries one or more
    physics-material schemas. The fragments author single namespaces onto that prim: the
    solver-common ``physics:*`` friction/restitution/density
    (:class:`UsdPhysicsRigidBodyMaterialCfg`) and any backend-specific namespace (e.g. PhysX
    ``physxMaterial:*`` via :class:`~isaaclab_physx.sim.spawners.materials.PhysxMaterialCfg`).
    The defining ``UsdPhysics.MaterialAPI`` anchor is applied by the family writer
    (:func:`~isaaclab.sim.spawners.materials.spawn_rigid_body_material_from_fragments`).
    """

    pass


@configclass
class UsdPhysicsRigidBodyMaterialCfg(RigidBodyMaterialFragment):
    """``physics:*`` rigid-body material attributes from `UsdPhysics.MaterialAPI`_.

    The ``UsdPhysics.MaterialAPI`` schema is applied as the implicit anchor by the rigid-body material
    family writer, so this fragment owns no applied schema of its own. ``None`` fields are left
    unchanged on the material prim.

    .. _UsdPhysics.MaterialAPI: https://openusd.org/dev/api/class_usd_physics_material_a_p_i.html
    """

    _usd_namespace: ClassVar[str | None] = "physics"
    _usd_applied_schema: ClassVar[str | None] = None  # MaterialAPI applied by the family writer

    static_friction: float | None = None
    """The static friction coefficient. Writes ``physics:staticFriction``."""

    dynamic_friction: float | None = None
    """The dynamic friction coefficient. Writes ``physics:dynamicFriction``."""

    restitution: float | None = None
    """The restitution coefficient. Writes ``physics:restitution``."""

    density: float | None = None
    """The material density [kg/m^3]. Writes ``physics:density``.

    Participates in mass computation via material binding when no explicit rigid-body mass or
    density takes precedence.
    """


@configclass
class DeformableBodyMaterialBaseCfg(PhysicsMaterialCfg):
    """Base physics material parameters for volume deformable bodies.

    Backend-specific subclasses provide the material fields and spawning function
    through :attr:`func`.
    """

    func: Callable | str | None = None


@configclass
class SurfaceDeformableBodyMaterialBaseCfg(DeformableBodyMaterialBaseCfg):
    """Base physics material parameters for surface deformable bodies."""
