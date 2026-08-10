# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Manager-based environment that installs the task-local conveyor force driver."""

from __future__ import annotations

from collections.abc import Sequence

from isaaclab.envs import ManagerBasedRLEnv

from .conveyor_force_driver import ConveyorForceDriver
from .conveyor_franka_env_cfg import ConveyorFrankaEnvCfg
from .conveyor_geometry import belt_collision_section_specs
from .conveyor_goal_selector import ConveyorGoalSelector


class ConveyorFrankaEnv(ManagerBasedRLEnv):
    """Manager-based environment with force-driven Newton conveyor surfaces."""

    cfg: ConveyorFrankaEnvCfg

    def __init__(self, cfg: ConveyorFrankaEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode=render_mode, **kwargs)
        self._conveyor_driver = ConveyorForceDriver(
            num_envs=self.num_envs,
            surface_specs=tuple(
                section for side in ("Left", "Right") for section in belt_collision_section_specs(side)
            ),
            speed=cfg.conveyor_force.speed,
            friction=cfg.conveyor_force.friction,
            normal_threshold=cfg.conveyor_force.normal_threshold,
            startup_duration_s=cfg.conveyor_force.startup_duration_s,
            transported_body_pattern=cfg.conveyor_force.transported_body_pattern,
            transported_body_count_per_env=cfg.conveyor_force.transported_body_count_per_env,
        )
        self._goal_selector: ConveyorGoalSelector | None = None
        self._setup_goal_selector()

    def _setup_goal_selector(self) -> None:
        """Attach one task panel to the first interactive Newton visualizer."""
        for visualizer in self.sim.visualizers:
            if getattr(visualizer.cfg, "visualizer_type", None) not in {"newton_gl", "newton_rtx"}:
                continue
            register_callback = getattr(visualizer, "register_ui_callback", None)
            if register_callback is None:
                continue
            visible_env_ids = visualizer.get_visualized_env_ids()
            env_id = visible_env_ids[0] if visible_env_ids else 0
            self._goal_selector = ConveyorGoalSelector(self, env_id)
            register_callback(self._goal_selector.render, position="panel")
            return

    def _reset_idx(self, env_ids: Sequence[int]):
        """Reset selected environments and discard stale conveyor forces."""
        super()._reset_idx(env_ids)

        conveyor_driver = getattr(self, "_conveyor_driver", None)
        if conveyor_driver is not None:
            conveyor_driver.reset(env_ids)

    def close(self):
        """Release the conveyor callbacks before the Newton scene is destroyed."""
        conveyor_driver = getattr(self, "_conveyor_driver", None)
        if conveyor_driver is not None:
            conveyor_driver.close()
            self._conveyor_driver = None
        super().close()
