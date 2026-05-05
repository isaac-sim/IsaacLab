# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration terms for different managers (experimental, Warp-first).

The warp manager classes accept the same term cfg shapes as the stable
managers, so this module simply re-exports the stable term cfg classes
to keep ``isinstance(stable_term, warp.TermCfg)`` true. This is what
lets the :class:`isaaclab_experimental.envs.warp_frontend.WarpFrontend`
adapter run a stable cfg through the warp runtime without rewrapping
every term.

The ``func`` callable on each term is still expected to follow the
warp-first ``func(env, out, **params) -> None`` signature when run on
the warp runtime; only the *type* is shared with stable.
"""

from __future__ import annotations

# Re-export stable manager term cfg classes verbatim.
# `from … import *` carries `ObservationTermCfg`, `RewardTermCfg`,
# `TerminationTermCfg`, `ManagerTermBaseCfg`, etc.
from isaaclab.managers.manager_term_cfg import *  # noqa: F401,F403
