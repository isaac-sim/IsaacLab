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
the ``requires_host_ids`` selector consumed by the Warp-first curriculum manager.
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

    requires_host_ids: bool | None = None
    """Whether a legacy term consumes compact environment IDs.

    ``None`` selects the manager default: mask-native terms do not require IDs, while legacy terms do.
    Set this to ``False`` for global legacy terms that ignore their ``env_ids`` argument.
    """


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
