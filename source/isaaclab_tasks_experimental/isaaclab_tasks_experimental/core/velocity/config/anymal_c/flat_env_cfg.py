# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Warp variants of the stable AnymalC flat velocity task configuration."""

from isaaclab.utils.configclass import configclass

from isaaclab_tasks.contrib.velocity.config.anymal_c.flat_env_cfg import AnymalCFlatEnvCfg as _StableAnymalCFlatEnvCfg
from isaaclab_tasks.contrib.velocity.config.anymal_c.flat_env_cfg import (
    AnymalCFlatEnvCfg_PLAY as _StableAnymalCFlatEnvCfg_PLAY,
)

from isaaclab_tasks_experimental.core.velocity import (
    disable_unsupported_randomization_events,
)


@configclass
class AnymalCFlatEnvCfg(_StableAnymalCFlatEnvCfg):
    """Stable flat cfg with the randomization events lacking warp twins disabled."""

    def __post_init__(self):
        super().__post_init__()
        disable_unsupported_randomization_events(self)


@configclass
class AnymalCFlatEnvCfg_PLAY(_StableAnymalCFlatEnvCfg_PLAY):
    """Play variant of :class:`AnymalCFlatEnvCfg` for the warp runtime."""

    def __post_init__(self):
        super().__post_init__()
        disable_unsupported_randomization_events(self)
