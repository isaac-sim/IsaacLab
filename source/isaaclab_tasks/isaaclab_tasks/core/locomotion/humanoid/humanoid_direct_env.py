# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from isaaclab_tasks.core.locomotion.humanoid.humanoid_direct_env_cfg import HumanoidEnvCfg
from isaaclab_tasks.core.locomotion.locomotion_direct_env import LocomotionDirectEnv


class HumanoidEnv(LocomotionDirectEnv):
    """Direct-workflow Humanoid locomotion environment.

    The behavior is fully defined by :class:`LocomotionDirectEnv` and :class:`HumanoidEnvCfg`.
    """

    cfg: HumanoidEnvCfg
