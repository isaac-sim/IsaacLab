# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Warp variants of the stable AnymalD flat velocity task configuration."""

from isaaclab.utils.configclass import configclass

from isaaclab_tasks.core.velocity.config.anymal_d.flat_env_cfg import AnymalDFlatEnvCfg as _StableAnymalDFlatEnvCfg
from isaaclab_tasks.core.velocity.config.anymal_d.flat_env_cfg import (
    AnymalDFlatEnvCfg_PLAY as _StableAnymalDFlatEnvCfg_PLAY,
)

from isaaclab_tasks_experimental.manager_based.locomotion.velocity import (
    disable_unsupported_randomization_events,
)


@configclass
class AnymalDFlatEnvCfg(_StableAnymalDFlatEnvCfg):
    """Stable flat cfg with the randomization events lacking warp twins disabled."""

    def __post_init__(self):
        super().__post_init__()
        disable_unsupported_randomization_events(self)


@configclass
class AnymalDFlatEnvCfg_PLAY(_StableAnymalDFlatEnvCfg_PLAY):
    """Play variant of :class:`AnymalDFlatEnvCfg` for the warp runtime."""

    def __post_init__(self):
        super().__post_init__()
        disable_unsupported_randomization_events(self)
