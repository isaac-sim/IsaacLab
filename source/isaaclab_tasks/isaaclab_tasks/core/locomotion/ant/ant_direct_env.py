# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Direct-workflow Ant locomotion environment."""

from __future__ import annotations

from ..locomotion_direct_env import LocomotionDirectEnv
from .ant_direct_env_cfg import AntEnvCfg


class AntEnv(LocomotionDirectEnv):
    """Direct-workflow Ant locomotion environment.

    The behavior is fully defined by :class:`LocomotionDirectEnv` and :class:`AntEnvCfg`.
    """

    cfg: AntEnvCfg
