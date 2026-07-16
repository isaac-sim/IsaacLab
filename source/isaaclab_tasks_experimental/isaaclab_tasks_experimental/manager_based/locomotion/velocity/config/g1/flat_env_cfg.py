# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Warp variants of the stable G1 flat velocity task configuration."""

from isaaclab.utils.configclass import configclass

from isaaclab_tasks.core.velocity.config.g1.flat_env_cfg import G1FlatEnvCfg as _StableG1FlatEnvCfg
from isaaclab_tasks.core.velocity.config.g1.flat_env_cfg import G1FlatEnvCfg_PLAY as _StableG1FlatEnvCfg_PLAY

from isaaclab_tasks_experimental.manager_based.locomotion.velocity import (
    disable_unsupported_randomization_events,
)


@configclass
class G1FlatEnvCfg(_StableG1FlatEnvCfg):
    """Stable flat cfg with the randomization events lacking warp twins disabled."""

    def __post_init__(self):
        super().__post_init__()
        disable_unsupported_randomization_events(self)


@configclass
class G1FlatEnvCfg_PLAY(_StableG1FlatEnvCfg_PLAY):
    """Play variant of :class:`G1FlatEnvCfg` for the warp runtime."""

    def __post_init__(self):
        super().__post_init__()
        disable_unsupported_randomization_events(self)
