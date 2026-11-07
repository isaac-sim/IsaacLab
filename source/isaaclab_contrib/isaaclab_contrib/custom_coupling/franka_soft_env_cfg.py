# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Deprecated Franka soft lifting task alias using generic proxy coupling."""

import warnings

from isaaclab.utils import configclass

from isaaclab_tasks.core.lift.config.franka_soft.franka_soft_env_cfg import FrankaSoftEnvCfg
from isaaclab_tasks.core.lift.config.franka_soft.franka_soft_env_cfg import PhysicsCfg as CorePhysicsCfg


@configclass
class PhysicsCfg(CorePhysicsCfg):
    """Keep the former manual preset name as an alias of the core proxy preset."""

    newton_mjwarp_vbd = CorePhysicsCfg().newton_mjwarp_vbd_proxy
    default = newton_mjwarp_vbd


@configclass
class FrankaSoftCustomCouplingEnvCfg(FrankaSoftEnvCfg):
    """Deprecated task alias for Franka soft lifting with generic proxy coupling."""

    def __post_init__(self) -> None:
        warnings.warn(
            "IsaacContrib-Lift-Soft-Franka-Custom-Coupling is deprecated and now uses generic proxy coupling. "
            "Use Isaac-Lift-Soft-Franka with physics=newton_mjwarp_vbd_proxy; retune former manual-coupling policies.",
            FutureWarning,
            stacklevel=2,
        )
        super().__post_init__()
        self.sim.physics = PhysicsCfg()
