# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Construction boundary for the contributed GR1T2 and G1 teleoperation tasks."""

from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg

from .teleop_physics_cfg import prepare_humanoid_teleop_environment


def make_humanoid_teleop_env(cfg: ManagerBasedRLEnvCfg, render_mode: str | None = None, **kwargs) -> ManagerBasedRLEnv:
    """Prepare the selected task backend before constructing the standard environment."""
    prepare_humanoid_teleop_environment(cfg)
    return ManagerBasedRLEnv(cfg, render_mode, **kwargs)
