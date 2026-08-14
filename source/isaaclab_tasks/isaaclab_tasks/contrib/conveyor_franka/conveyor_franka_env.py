# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Manager-based environment that installs the selected conveyor physics adapter."""

from __future__ import annotations

from collections.abc import Sequence

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.physics import ConveyorBeltView

from .conveyor_force_driver import ConveyorForceDriver
from .conveyor_franka_env_cfg import ConveyorFrankaEnvCfg
from .conveyor_geometry import belt_collision_section_specs
from .conveyor_goal_selector import ConveyorGoalSelector
from .conveyor_physx_surface import PhysxSurfaceVelocityConveyor


class ConveyorFrankaEnv(ManagerBasedRLEnv):
    """Manager-based environment with backend-native conveyor surfaces."""

    cfg: ConveyorFrankaEnvCfg

    def __init__(self, cfg: ConveyorFrankaEnvCfg, render_mode: str | None = None, **kwargs):
        self._conveyor_driver: ConveyorBeltView | None = None
        super().__init__(cfg, render_mode=render_mode, **kwargs)
        self._goal_selector: ConveyorGoalSelector | None = None
        self._setup_goal_selector()

    def _init_sim(self) -> None:
        """Install the conveyor adapter at the lifecycle point required by its backend."""
        belt_spec_kwargs = {
            "velocity": self.cfg.conveyor_force.speed,
            "friction_coefficient": self.cfg.conveyor_force.friction,
            "contact_threshold": self.cfg.conveyor_force.normal_threshold,
        }
        spec_builder = getattr(self.cfg.scene, "build_conveyor_belt_specs", None)
        if spec_builder is None:
            belt_specs = tuple(
                section.belt
                for side in ("Left", "Right")
                for section in belt_collision_section_specs(side, **belt_spec_kwargs)
            )
        else:
            belt_specs = tuple(spec_builder(**belt_spec_kwargs))
        env_path_format = self.cfg.scene.clone_cfg.clone_template

        # Newton needs solved-contact attributes and graph callbacks registered
        # before the first reset finalizes and captures the solver. PhysX belt
        # schemas, by contrast, are authored by the scene spawners and its live
        # command adapter is attached only after PhysX has parsed that scene.
        from isaaclab_newton.physics import NewtonCfg

        if isinstance(self.cfg.sim.physics, NewtonCfg):
            driver = ConveyorForceDriver(
                num_envs=self.cfg.scene.num_envs,
                belt_specs=belt_specs,
                startup_duration_s=self.cfg.conveyor_force.startup_duration_s,
                env_path_format=env_path_format,
                transported_body_pattern=self.cfg.conveyor_force.transported_body_pattern,
                transported_body_count_per_env=self.cfg.conveyor_force.transported_body_count_per_env,
            )
            self._conveyor_driver = driver
            try:
                super()._init_sim()
            except Exception:
                driver.close()
                self._conveyor_driver = None
                raise
            return

        from isaaclab_physx.physics import PhysxCfg

        if not isinstance(self.cfg.sim.physics, PhysxCfg):
            raise ValueError(f"Unsupported conveyor physics backend: {type(self.cfg.sim.physics).__name__}.")

        configure_conveyor = getattr(self.cfg.scene, "configure_conveyor", None)
        if configure_conveyor is not None:
            configure_conveyor(friction_coefficient=self.cfg.conveyor_force.friction)
        super()._init_sim()
        driver = PhysxSurfaceVelocityConveyor(
            num_envs=self.cfg.scene.num_envs,
            belt_specs=belt_specs,
            env_path_format=env_path_format,
            startup_duration_s=self.cfg.conveyor_force.startup_duration_s,
            stage=self.sim.stage,
        )
        try:
            driver.start()
        except Exception:
            driver.close()
            raise
        self._conveyor_driver = driver

    @property
    def conveyor_belt(self) -> ConveyorBeltView:
        """Tensorized conveyor control view for this environment."""
        driver = self._conveyor_driver
        if driver is None:
            raise RuntimeError("The conveyor belt is unavailable before simulation initialization or after close().")
        return driver

    def _setup_goal_selector(self) -> None:
        """Attach one task panel to the first supported interactive visualizer."""
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
        """Release conveyor callbacks before the physics scene is destroyed."""
        conveyor_driver = getattr(self, "_conveyor_driver", None)
        if conveyor_driver is not None:
            conveyor_driver.close()
            self._conveyor_driver = None
        super().close()
