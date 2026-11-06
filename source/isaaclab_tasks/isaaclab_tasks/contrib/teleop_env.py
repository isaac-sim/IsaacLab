# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Construction boundary for the contributed GR1T2 and G1 teleoperation tasks."""

from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg

from .teleop_physics_cfg import prepare_humanoid_teleop_environment


class HumanoidTeleopEnv(ManagerBasedRLEnv):
    """Prepare the selected task backend before physics and scene construction."""

    def __init__(self, cfg: ManagerBasedRLEnvCfg, render_mode: str | None = None, **kwargs):
        prepare_humanoid_teleop_environment(cfg)
        super().__init__(cfg, render_mode, **kwargs)
