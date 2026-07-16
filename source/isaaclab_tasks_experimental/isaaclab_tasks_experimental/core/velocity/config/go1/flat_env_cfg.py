# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Warp variants of the stable Unitree Go1 flat velocity task configuration."""

from isaaclab.utils.configclass import configclass

from isaaclab_tasks.contrib.velocity.config.go1.flat_env_cfg import UnitreeGo1FlatEnvCfg as _StableUnitreeGo1FlatEnvCfg
from isaaclab_tasks.contrib.velocity.config.go1.flat_env_cfg import (
    UnitreeGo1FlatEnvCfg_PLAY as _StableUnitreeGo1FlatEnvCfg_PLAY,
)

from isaaclab_tasks_experimental.core.velocity import (
    disable_unsupported_randomization_events,
)


@configclass
class UnitreeGo1FlatEnvCfg(_StableUnitreeGo1FlatEnvCfg):
    """Stable flat cfg with the randomization events lacking warp twins disabled."""

    def __post_init__(self):
        super().__post_init__()
        disable_unsupported_randomization_events(self)


@configclass
class UnitreeGo1FlatEnvCfg_PLAY(_StableUnitreeGo1FlatEnvCfg_PLAY):
    """Play variant of :class:`UnitreeGo1FlatEnvCfg` for the warp runtime."""

    def __post_init__(self):
        super().__post_init__()
        disable_unsupported_randomization_events(self)
