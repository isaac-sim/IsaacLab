# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Manager term configuration classes used by the Warp-first managers.

Passthrough to :mod:`isaaclab.managers.manager_term_cfg`: the Warp-first managers
accept the same configuration shapes as the stable managers, so re-exporting the
stable classes preserves type identity and lets a stable task configuration be
adapted in place (by the warp frontend) without rebuilding every term. At
runtime, the adapted term callables still use the Warp-first
``func(env, out, **params) -> None`` signature.

:class:`CurriculumTermCfg` is the one override: it extends the stable class with
the Warp-first curriculum term configuration.
"""

from __future__ import annotations

from isaaclab.managers.manager_term_cfg import (
    ActionTermCfg,
    CommandTermCfg,
    EventTermCfg,
    ManagerTermBaseCfg,
    ObservationGroupCfg,
    ObservationTermCfg,
    RecorderTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
)
from isaaclab.managers.manager_term_cfg import CurriculumTermCfg as _CurriculumTermCfg
from isaaclab.utils.configclass import configclass


@configclass
class CurriculumTermCfg(_CurriculumTermCfg):
    """Configuration for a Warp-mask or legacy curriculum term."""


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
