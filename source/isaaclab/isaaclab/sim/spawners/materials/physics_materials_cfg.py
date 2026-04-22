# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from dataclasses import MISSING
from typing import Literal

from isaaclab.utils import configclass


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
class RigidBodyMaterialCfg(PhysicsMaterialCfg):
    """Physics material parameters for rigid bodies.

    See :meth:`spawn_rigid_body_material` for more information.
    """

    func: Callable | str = "{DIR}.physics_materials:spawn_rigid_body_material"

    static_friction: float = 0.5
    """The static friction coefficient. Defaults to 0.5."""

    dynamic_friction: float = 0.5
    """The dynamic friction coefficient. Defaults to 0.5."""

    restitution: float = 0.0
    """The restitution coefficient. Defaults to 0.0."""

    friction_combine_mode: Literal["average", "min", "multiply", "max"] = "average"
    """Determines the way friction will be combined during collisions. Defaults to `"average"`.

    .. attention::

        When two physics materials with different combine modes collide, the combine mode with the higher
        priority will be used. The priority order is provided `here
        <https://nvidia-omniverse.github.io/PhysX/physx/5.4.1/_api_build/structPxCombineMode.html>`__.
    """

    restitution_combine_mode: Literal["average", "min", "multiply", "max"] = "average"
    """Determines the way restitution coefficient will be combined during collisions. Defaults to `"average"`.

    .. attention::

        When two physics materials with different combine modes collide, the combine mode with the higher
        priority will be used. The priority order is provided `here
        <https://nvidia-omniverse.github.io/PhysX/physx/5.4.1/_api_build/structPxCombineMode.html>`__.
    """

    compliant_contact_stiffness: float = 0.0
    """Spring stiffness for a compliant contact model using implicit springs. Defaults to 0.0.

    A higher stiffness results in behavior closer to a rigid contact. The compliant contact model is only enabled
    if the stiffness is larger than 0.
    """

    compliant_contact_damping: float = 0.0
    """Damping coefficient for a compliant contact model using implicit springs. Defaults to 0.0.

    Irrelevant if compliant contacts are disabled when :obj:`compliant_contact_stiffness` is set to zero and
    rigid contacts are active.
    """


@configclass
class OmniPhysicsDeformableMaterialCfg:
    """OmniPhysics material properties for a deformable body.

    These properties are set with the prefix ``omniphysics:<property_name>``. For example, to set the density of the
    deformable body, you would set the property ``omniphysics:density``.

    See the OmniPhysics documentation for more information on the available properties.
    """

    density: float = 1000.0
    """The material density in [kg/m^3]. Defaults to 1000.0 kg/m^3, which is the density of water."""

    static_friction: float = 0.25
    """The static friction. Defaults to 0.25."""

    dynamic_friction: float = 0.25
    """The dynamic friction. Defaults to 0.25."""

    youngs_modulus: float = 1000000.0
    """The Young's modulus, which defines the body's stiffness. Defaults to 1[MPa].

    The Young's modulus is a measure of the material's ability to deform under stress. It is measured in Pascals ([Pa]).
    """

    poissons_ratio: float = 0.45
    """The Poisson's ratio which defines the body's volume preservation. Defaults to 0.45.

    The Poisson's ratio is a measure of the material's ability to expand in the lateral direction when compressed
    in the axial direction. It is a dimensionless number between 0 and 0.5. Using a value of 0.5 will make the
    material incompressible.
    """


@configclass
class OmniPhysicsSurfaceDeformableMaterialCfg(OmniPhysicsDeformableMaterialCfg):
    """OmniPhysics material properties for a surface deformable body,
    extending on :class:`OmniPhysicsDeformableMaterialCfg` with additional parameters for surface deformable bodies.

    These properties are set with the prefix ``omniphysics:<property_name>``.
    For example, to set the surface thickness of the surface deformable body,
    you would set the property ``omniphysics:surfaceThickness``.

    See the OmniPhysics documentation for more information on the available properties.
    """

    surface_thickness: float = 0.01
    """The thickness of the deformable body's surface. Defaults to 0.01 meters ([m])."""

    surface_stretch_stiffness: float = 0.0
    """The stretch stiffness of the deformable body's surface. Defaults to 0.0."""

    surface_shear_stiffness: float = 0.0
    """The shear stiffness of the deformable body's surface. Defaults to 0.0."""

    surface_bend_stiffness: float = 0.0
    """The bend stiffness of the deformable body's surface. Defaults to 0.0."""

    bend_damping: float = 0.0
    """The bend damping for the deformable body's surface. Defaults to 0.0."""


@configclass
class PhysXDeformableMaterialCfg:
    """PhysX-specific material properties for a deformable body.

    These properties are set with the prefix ``physxDeformableBody:<property_name>``.
    For example, to set the elasticity damping of the deformable body,
    you would set the property ``physxDeformableBody:elasticityDamping``.

    See the PhysX documentation for more information on the available properties.
    """

    elasticity_damping: float = 0.005
    """The elasticity damping for the deformable material. Defaults to 0.005."""


@configclass
class NewtonDeformableMaterialCfg:
    """Newton-specific material properties for a deformable body.

    These properties are set with the prefix ``newton:<property_name>``.
    For example, to set the elasticity damping of the deformable body,
    you would set the property ``newton:tri_ke``.

    See the Newton documentation for more information on the available properties.
    """

    particle_radius: float = 0.008
    """Particle radius [m] (controls rigid body-particle contact distance). Used by Newton backend only."""

    # -- Cloth (triangle surface mesh) parameters

    tri_ke: float = 1e4
    """Triangle area-preserving stiffness [Pa]. Used by Newton backend for cloth meshes."""

    tri_ka: float = 1e4
    """Triangle area stiffness [Pa]. Used by Newton backend for cloth meshes."""

    tri_kd: float = 1.5e-6
    """Triangle area damping. Used by Newton backend for cloth meshes."""

    edge_ke: float = 5.0
    """Bending stiffness. Used by Newton backend for cloth meshes."""

    edge_kd: float = 1e-2
    """Bending damping. Used by Newton backend for cloth meshes."""

    # -- Volumetric (tetrahedral FEM) parameters

    k_damp: float = 0.0
    """Damping stiffness for tetrahedral elements. Defaults to 0.0."""


@configclass
class DeformableBodyMaterialCfg(
    PhysicsMaterialCfg,
    OmniPhysicsDeformableMaterialCfg,
    PhysXDeformableMaterialCfg,
    NewtonDeformableMaterialCfg,
):
    """Physics material parameters for deformable bodies.

    See :meth:`spawn_deformable_body_material` for more information.
    """

    func: Callable | str = "{DIR}.physics_materials:spawn_deformable_body_material"

    _property_prefix: dict[str, list[str]] = {
        "omniphysics": [field.name for field in dataclasses.fields(OmniPhysicsDeformableMaterialCfg)],
        "physxDeformableBody": [field.name for field in dataclasses.fields(PhysXDeformableMaterialCfg)],
        "newton": [field.name for field in dataclasses.fields(NewtonDeformableMaterialCfg)],
    }
    """Mapping between the property prefixes and the properties that fall under each prefix."""


@configclass
class SurfaceDeformableBodyMaterialCfg(DeformableBodyMaterialCfg, OmniPhysicsSurfaceDeformableMaterialCfg):
    """Physics material parameters for surface deformable bodies,
    extending on :class:`DeformableBodyMaterialCfg` with additional parameters for surface deformable bodies.

    See :meth:`spawn_deformable_body_material` for more information.
    """

    func: Callable | str = "{DIR}.physics_materials:spawn_deformable_body_material"

    _property_prefix: dict[str, list[str]] = {
        "omniphysics": [field.name for field in dataclasses.fields(OmniPhysicsSurfaceDeformableMaterialCfg)],
        "physxDeformableBody": [field.name for field in dataclasses.fields(PhysXDeformableMaterialCfg)],
        "newton": [field.name for field in dataclasses.fields(NewtonDeformableMaterialCfg)],
    }
    """Extend DeformableBodyMaterialCfg properties under each prefix."""
