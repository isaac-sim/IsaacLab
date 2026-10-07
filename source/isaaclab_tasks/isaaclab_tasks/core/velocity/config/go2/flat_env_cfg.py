# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Unitree Go2 velocity-tracking environment on flat terrain."""

from isaaclab_newton.physics import FeatherPGSSolverCfg, NewtonCfg, NewtonCollisionPipelineCfg, NewtonShapeCfg

from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from ...velocity_env_cfg import RoughPhysicsCfg
from .rough_env_cfg import UnitreeGo2RoughEnvCfg


@configclass
class UnitreeGo2FlatPhysicsCfg(RoughPhysicsCfg):
    """Physics backend presets for the Unitree Go2 velocity-tracking environment on flat terrain."""

    feather_pgs: NewtonCfg = NewtonCfg(
        solver_cfg=FeatherPGSSolverCfg(
            enable_joint_limits=True,
            joint_limit_activation_gap=0.2,
            pgs_iterations=8,
            dense_max_constraints=96,
            mf_max_constraints=64,
            pgs_beta=0.05,
        ),
        collision_cfg=NewtonCollisionPipelineCfg(rigid_contacts_per_world=16),
        default_shape_cfg=NewtonShapeCfg(gap=0.003),
        num_substeps=1,
        debug_mode=False,
        use_cuda_graph=True,
    )


@configclass
class UnitreeGo2FlatEnvCfg(UnitreeGo2RoughEnvCfg):
    """Configuration for the Unitree Go2 velocity-tracking environment on flat terrain."""

    sim: SimulationCfg = SimulationCfg(physics=UnitreeGo2FlatPhysicsCfg())

    def __post_init__(self):
        super().__post_init__()

        # physics
        newton_mjwarp = self.sim.physics.newton_mjwarp
        newton_mjwarp.solver_cfg.njmax = 65
        newton_mjwarp.solver_cfg.nconmax = 35
        self.sim.physics.default = newton_mjwarp
        # scene
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        # observations
        self.observations.policy.height_scan = None
        # rewards
        self.rewards.flat_orientation_l2.weight = -2.5
        self.rewards.feet_air_time.weight = 0.25
        self.rewards.base_height_l2.params["sensor_cfg"] = None
        # curriculum
        self.curriculum.terrain_levels = None
