# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.sim import SimulationCfg
from isaaclab.sim._impl.newton_manager_cfg import NewtonCfg
from isaaclab.sim._impl.solvers_cfg import MJWarpSolverCfg
from isaaclab.utils import configclass

from .rough_env_cfg import AnymalBRoughEnvCfg


@configclass
class AnymalBFlatEnvCfg(AnymalBRoughEnvCfg):
    sim: SimulationCfg = SimulationCfg(
        newton_cfg=NewtonCfg(
            solver_cfg=MJWarpSolverCfg(
                njmax=50,
                nconmax=15,
                ls_iterations=15,
                cone="elliptic",
                impratio=100,
                ls_parallel=True,
                integrator="implicitfast",
            ),
            num_substeps=1,
            debug_mode=False,
        )
    )

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # override rewards
        self.rewards.flat_orientation_l2.weight = -5.0
        self.rewards.dof_torques_l2.weight = -2.5e-5
        self.rewards.feet_air_time.weight = 0.5
        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        # self.scene.height_scanner = None
        # self.observations.policy.height_scan = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None


class AnymalBFlatEnvCfg_PLAY(AnymalBFlatEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None
