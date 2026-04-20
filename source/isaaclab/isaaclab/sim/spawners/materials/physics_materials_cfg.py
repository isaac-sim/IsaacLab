# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

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
