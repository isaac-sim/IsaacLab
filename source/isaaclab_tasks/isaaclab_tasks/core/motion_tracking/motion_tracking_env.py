# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Motion tracking with contact history sampled at the physics rate."""

from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg


class MotionTrackingEnv(ManagerBasedRLEnv):
    """Preserve each physics step for the self-collision reward."""

    def __init__(self, cfg: ManagerBasedRLEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode=render_mode, **kwargs)
        # Filtered contacts are sampled by scene.update, once per physics step.
        self.sim.backend_manager.set_decimation(1)
        self._physics_handles_decimation = False
