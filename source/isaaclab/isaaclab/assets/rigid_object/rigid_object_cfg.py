# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from typing import TYPE_CHECKING

from ...utils import configclass
from ..asset_base_cfg import AssetBaseCfg

if TYPE_CHECKING:
    from .rigid_object import RigidObject


@configclass
class RigidObjectCfg(AssetBaseCfg):
    """Configuration parameters for a rigid object."""

    @configclass
    class InitialStateCfg(AssetBaseCfg.InitialStateCfg):
        """Initial state of the rigid body."""

        lin_vel: tuple[float, float, float] = (0.0, 0.0, 0.0)
        """Linear velocity of the root in simulation world frame. Defaults to (0.0, 0.0, 0.0)."""
        ang_vel: tuple[float, float, float] = (0.0, 0.0, 0.0)
        """Angular velocity of the root in simulation world frame. Defaults to (0.0, 0.0, 0.0)."""

    ##
    # Initialize configurations.
    ##

    class_type: type["RigidObject"] | str = "{DIR}.rigid_object:RigidObject"

    init_state: InitialStateCfg = InitialStateCfg()
    """Initial state of the rigid object. Defaults to identity pose with zero velocity."""

    mesh_variants_enabled: bool = False
    """Prepare multi-asset collision meshes for reset-time selection.

    Newton MJWarp requires one rigid body and the same number of mesh colliders in every variant.
    Clone planning authors every variant's source so the model owns its mesh and SDF resources,
    including variants not initially selected by any environment. Selection updates collision geometry,
    mass, and inertia. Newton OpenGL and Viser visualizers follow the selection without rewriting USD.
    Selection requires Newton external contacts and disabled sleeping; MuJoCo's compiled meshes are
    not updated. Internal MuJoCo contacts and geometry-dependent MuJoCo queries are unsupported.
    PhysX ignores this option; Newton solvers other than MJWarp reject it. Defaults to ``False``.
    """

    mesh_variant_inertia_diagonal_offset: float = 0.0
    """Diagonal inertia added to every mesh variant [kg·m²]. Defaults to ``0.0``."""
