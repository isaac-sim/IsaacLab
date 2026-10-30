# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Native CPU PhysX initialization for the shared conveyor task."""

from isaaclab.envs import ManagerBasedRLEnv

from isaaclab_contrib.conveyors.physx import SurfaceVelocity

from .conveyor_franka_env import ConveyorFrankaEnv
from .conveyor_franka_physx_env_cfg import ConveyorFrankaPhysxEnvCfg


class ConveyorFrankaPhysxEnv(ConveyorFrankaEnv):
    """Reuse the transfer task with PhysX-authored moving surfaces."""

    cfg: ConveyorFrankaPhysxEnvCfg

    def _init_sim(self) -> None:
        """Parse the authored PhysX scene before binding its live surface commands."""
        self.cfg.scene.configure_conveyor(friction_coefficient=self.cfg.conveyor_force.friction)
        ManagerBasedRLEnv._init_sim(self)
        driver = SurfaceVelocity(
            num_envs=self.cfg.scene.num_envs,
            surface_specs=self.cfg.scene.build_conveyor_belt_specs(
                velocity=self.cfg.conveyor_force.speed,
                friction_coefficient=self.cfg.conveyor_force.friction,
                contact_threshold=self.cfg.conveyor_force.normal_threshold,
            ),
            env_path_format=self.cfg.scene.clone_cfg.clone_template,
            startup_duration_s=self.cfg.conveyor_force.startup_duration_s,
            stage=self.sim.stage,
            manager=self.sim.backend_manager,
        )
        try:
            driver.start()
        except Exception:
            driver.close()
            raise
        self._conveyor_driver = driver
