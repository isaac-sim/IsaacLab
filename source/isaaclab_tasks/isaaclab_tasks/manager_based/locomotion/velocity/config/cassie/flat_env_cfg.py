# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg

from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from .rough_env_cfg import CassieRoughEnvCfg


@configclass
class CassieFlatEnvCfg(CassieRoughEnvCfg):
    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 200.0,
        physics=NewtonCfg(
            solver_cfg=MJWarpSolverCfg(
                njmax=52,
                nconmax=15,
                ls_iterations=10,
                cone="pyramidal",
                impratio=1,
                ls_parallel=True,
                integrator="implicitfast",
            ),
            num_substeps=1,
            debug_mode=False,
        ),
    )

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # rewards
        self.rewards.flat_orientation_l2.weight = -2.5
        self.rewards.feet_air_time.weight = 5.0
        self.rewards.joint_deviation_hip.params["asset_cfg"].joint_names = ["hip_rotation_.*"]
        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        # self.scene.height_scanner = None
        # self.observations.policy.height_scan = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None


class CassieFlatEnvCfg_PLAY(CassieFlatEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
