# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Warp variants of the stable Unitree Go2 flat velocity task configuration."""

from isaaclab.utils.configclass import configclass

from isaaclab_tasks.core.velocity.config.go2.flat_env_cfg import UnitreeGo2FlatEnvCfg as _StableUnitreeGo2FlatEnvCfg
from isaaclab_tasks.core.velocity.config.go2.flat_env_cfg import (
    UnitreeGo2FlatEnvCfg_PLAY as _StableUnitreeGo2FlatEnvCfg_PLAY,
)

from isaaclab_tasks_experimental.core.velocity import (
    disable_unsupported_randomization_events,
)


@configclass
class UnitreeGo2FlatEnvCfg(_StableUnitreeGo2FlatEnvCfg):
    """Stable flat cfg with the randomization events lacking warp twins disabled."""

    def __post_init__(self):
        super().__post_init__()
        disable_unsupported_randomization_events(self)


@configclass
class UnitreeGo2FlatEnvCfg_PLAY(_StableUnitreeGo2FlatEnvCfg_PLAY):
    """Play variant of :class:`UnitreeGo2FlatEnvCfg` for the warp runtime."""

    def __post_init__(self):
        super().__post_init__()
        disable_unsupported_randomization_events(self)
