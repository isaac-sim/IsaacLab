# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Manager-based conveyor environment with Newton contact forces."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.envs import ManagerBasedRLEnv

from isaaclab_contrib.conveyors.newton import SurfaceVelocity

from .conveyor_franka_env_cfg import ConveyorFrankaEnvCfg
from .conveyor_goal_selector import ConveyorGoalSelector


class ConveyorFrankaEnv(ManagerBasedRLEnv):
    """Manager-based environment with Newton conveyor surfaces."""

    cfg: ConveyorFrankaEnvCfg

    def __init__(self, cfg: ConveyorFrankaEnvCfg, render_mode: str | None = None, **kwargs) -> None:
        self._conveyor_driver: SurfaceVelocity | None = None
        super().__init__(cfg, render_mode=render_mode, **kwargs)
        self._setup_goal_selector()

    def _init_sim(self) -> None:
        """Register solved-contact hooks before the first reset captures the solver."""
        driver = SurfaceVelocity(
            manager=self.sim.backend_manager,
            num_envs=self.cfg.scene.num_envs,
            surface_specs=self.cfg.scene.build_conveyor_belt_specs(
                velocity=self.cfg.conveyor_force.speed,
                friction_coefficient=self.cfg.conveyor_force.friction,
                contact_threshold=self.cfg.conveyor_force.normal_threshold,
            ),
            startup_duration_s=self.cfg.conveyor_force.startup_duration_s,
            env_path_format=self.cfg.scene.clone_cfg.clone_template,
            body_pattern=self.cfg.conveyor_force.transported_body_pattern,
            body_count_per_env=self.cfg.conveyor_force.transported_body_count_per_env,
        )
        self._conveyor_driver = driver
        try:
            super()._init_sim()
        except Exception:
            driver.close()
            self._conveyor_driver = None
            raise

    def load_managers(self) -> None:
        """Bind the fixed Panda tool frame before manager terms inspect it."""
        hand_ids, _ = self.scene["robot"].find_bodies("panda_hand")
        if len(hand_ids) != 1:
            raise ValueError(f"Expected one Panda hand body, found {len(hand_ids)}.")
        self._conveyor_hand_body_id = hand_ids[0]
        self._conveyor_tool_offset = torch.tensor((0.0, 0.0, 0.1034), dtype=torch.float32, device=self.device).expand(
            self.num_envs, -1
        )
        super().load_managers()

    def _setup_goal_selector(self) -> None:
        """Attach one task panel to the first GL visualizer."""
        for visualizer in self.sim.visualizers:
            if visualizer.cfg.backend.visualizer_type != "newton_gl":
                continue
            visible_env_ids = visualizer.get_visualized_env_ids()
            env_id = visible_env_ids[0] if visible_env_ids else 0
            selector = ConveyorGoalSelector(self, env_id)
            visualizer.register_ui_callback(selector.render, position="panel")
            return

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | slice) -> None:
        """Reset selected environments and discard stale conveyor forces."""
        if isinstance(env_ids, slice):
            env_ids = self.scene._ALL_INDICES[env_ids]
        super()._reset_idx(env_ids)

        conveyor_driver = self._conveyor_driver
        if conveyor_driver is not None:
            conveyor_driver.reset(env_ids)

    def close(self) -> None:
        """Release conveyor callbacks before the physics scene is destroyed."""
        conveyor_driver = self._conveyor_driver
        if conveyor_driver is not None:
            conveyor_driver.close()
            self._conveyor_driver = None
        super().close()
