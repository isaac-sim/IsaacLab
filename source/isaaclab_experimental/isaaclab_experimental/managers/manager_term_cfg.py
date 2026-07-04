# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Stable manager term configuration classes used by the Warp-first managers.

The Warp manager implementations accept the same configuration shapes as the
stable managers. Re-exporting the stable classes preserves type identity so a
stable task configuration can be adapted in place without rebuilding every
term. At runtime, the adapted term callables still use the Warp-first
``func(env, out, **params) -> None`` signature.
"""

from isaaclab.managers.manager_term_cfg import (
    ActionTermCfg,
    CommandTermCfg,
    CurriculumTermCfg,
    EventTermCfg,
    ManagerTermBaseCfg,
    ObservationGroupCfg,
    ObservationTermCfg,
    RecorderTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
)

__all__ = [
    "ActionTermCfg",
    "CommandTermCfg",
    "CurriculumTermCfg",
    "EventTermCfg",
    "ManagerTermBaseCfg",
    "ObservationGroupCfg",
    "ObservationTermCfg",
    "RecorderTermCfg",
    "RewardTermCfg",
    "TerminationTermCfg",
]
