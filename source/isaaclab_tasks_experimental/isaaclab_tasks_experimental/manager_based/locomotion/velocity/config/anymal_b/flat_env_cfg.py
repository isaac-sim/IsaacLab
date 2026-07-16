# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Warp variants of the stable AnymalB flat velocity task configuration."""

from isaaclab.utils.configclass import configclass

from isaaclab_tasks.contrib.velocity.config.anymal_b.flat_env_cfg import AnymalBFlatEnvCfg as _StableAnymalBFlatEnvCfg
from isaaclab_tasks.contrib.velocity.config.anymal_b.flat_env_cfg import (
    AnymalBFlatEnvCfg_PLAY as _StableAnymalBFlatEnvCfg_PLAY,
)

from isaaclab_tasks_experimental.manager_based.locomotion.velocity import (
    disable_unsupported_randomization_events,
)


@configclass
class AnymalBFlatEnvCfg(_StableAnymalBFlatEnvCfg):
    """Stable flat cfg with the randomization events lacking warp twins disabled."""

    def __post_init__(self):
        super().__post_init__()
        disable_unsupported_randomization_events(self)


@configclass
class AnymalBFlatEnvCfg_PLAY(_StableAnymalBFlatEnvCfg_PLAY):
    """Play variant of :class:`AnymalBFlatEnvCfg` for the warp runtime."""

    def __post_init__(self):
        super().__post_init__()
        disable_unsupported_randomization_events(self)
