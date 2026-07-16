# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Warp variants of the stable Cassie flat velocity task configuration."""

from isaaclab.utils.configclass import configclass

from isaaclab_tasks.core.velocity.config.cassie.flat_env_cfg import CassieFlatEnvCfg as _StableCassieFlatEnvCfg
from isaaclab_tasks.core.velocity.config.cassie.flat_env_cfg import (
    CassieFlatEnvCfg_PLAY as _StableCassieFlatEnvCfg_PLAY,
)

from isaaclab_tasks_experimental.manager_based.locomotion.velocity import (
    disable_unsupported_randomization_events,
)


@configclass
class CassieFlatEnvCfg(_StableCassieFlatEnvCfg):
    """Stable flat cfg with the randomization events lacking warp twins disabled."""

    def __post_init__(self):
        super().__post_init__()
        disable_unsupported_randomization_events(self)


@configclass
class CassieFlatEnvCfg_PLAY(_StableCassieFlatEnvCfg_PLAY):
    """Play variant of :class:`CassieFlatEnvCfg` for the warp runtime."""

    def __post_init__(self):
        super().__post_init__()
        disable_unsupported_randomization_events(self)
