# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Callable
from dataclasses import MISSING

from isaaclab.utils import configclass

# Names that moved out of this submodule into ``isaaclab_physx.sim.spawners.materials.physics_materials_cfg``.
# Resolved lazily so callers using ``from isaaclab.sim.spawners.materials.physics_materials_cfg
# import RigidBodyMaterialCfg`` continue to work without importing ``isaaclab_physx`` at module
# load time.
_PHYSX_FORWARDS = frozenset({"RigidBodyMaterialCfg", "PhysxRigidBodyMaterialCfg"})


def __getattr__(name):
    if name in _PHYSX_FORWARDS:
        try:
            from isaaclab_physx.sim.spawners.materials import physics_materials_cfg as _physx_mat_cfg
        except ImportError as e:
            raise ImportError(
                f"'isaaclab.sim.spawners.materials.physics_materials_cfg.{name}' has moved to"
                " 'isaaclab_physx.sim.spawners.materials.physics_materials_cfg'. Install the"
                " isaaclab_physx extension or update your import. This forwarding shim is scheduled"
                " for removal in 5.0."
            ) from e
        return getattr(_physx_mat_cfg, name)
    raise AttributeError(f"module 'isaaclab.sim.spawners.materials.physics_materials_cfg' has no attribute {name!r}")


@configclass
class PhysicsMaterialCfg:
    """Configuration parameters for creating a physics material.

    Physics material are PhysX schemas that can be applied to a USD material prim to define the
    physical properties related to the material. For example, the friction coefficient, restitution
    coefficient, etc. For more information on physics material, please refer to the
    `PhysX documentation <https://nvidia-omniverse.github.io/PhysX/physx/5.4.1/_api_build/classPxBaseMaterial.html>`__.
    """

    func: Callable = MISSING
    """Function to use for creating the material."""


@configclass
class RigidBodyMaterialBaseCfg(PhysicsMaterialCfg):
    """Solver-common physics-material parameters for rigid bodies.

    Contains the friction and restitution fields from the `UsdPhysics.MaterialAPI`_ that are common
    across all simulation backends. For PhysX-only material properties (compliant-contact spring,
    combine modes), use :class:`~isaaclab_physx.sim.spawners.materials.PhysxRigidBodyMaterialCfg`.

    See :meth:`spawn_rigid_body_material` for more information.

    .. _UsdPhysics.MaterialAPI: https://openusd.org/dev/api/class_usd_physics_material_a_p_i.html
    """

    func: Callable | str = "{DIR}.physics_materials:spawn_rigid_body_material"

    static_friction: float = 0.5
    """The static friction coefficient. Defaults to 0.5."""

    dynamic_friction: float = 0.5
    """The dynamic friction coefficient. Defaults to 0.5."""

    restitution: float = 0.0
    """The restitution coefficient. Defaults to 0.0."""
