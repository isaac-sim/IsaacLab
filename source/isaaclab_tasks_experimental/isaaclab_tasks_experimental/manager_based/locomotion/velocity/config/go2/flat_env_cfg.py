# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg

from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from .rough_env_cfg import UnitreeGo2RoughEnvCfg


@configclass
class UnitreeGo2FlatEnvCfg(UnitreeGo2RoughEnvCfg):
    sim: SimulationCfg = SimulationCfg(
        physics=NewtonCfg(
            solver_cfg=MJWarpSolverCfg(
                njmax=65,
                nconmax=35,
                cone="pyramidal",
                impratio=1,
                integrator="implicitfast",
            ),
            num_substeps=1,
            debug_mode=False,
        )
    )

    def __post_init__(self):
        super().__post_init__()
        self.rewards.flat_orientation_l2.weight = -2.5
        self.rewards.feet_air_time.weight = 0.25
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.curriculum.terrain_levels = None


class UnitreeGo2FlatEnvCfg_PLAY(UnitreeGo2FlatEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
