# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_newton.physics import (
    FeatherPGSSolverCfg,
    KaminoSolverCfg,
    MJWarpSolverCfg,
    NewtonCfg,
    NewtonCollisionPipelineCfg,
    NewtonShapeCfg,
)
from isaaclab_physx.physics import PhysxCfg

from isaaclab.sim import SimulationCfg
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.utils import PresetCfg

from .rough_env_cfg import CassieRoughEnvCfg


@configclass
class PhysicsCfg(PresetCfg):
    default = PhysxCfg(gpu_max_rigid_patch_count=10 * 2**15)
    newton_mjwarp = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            njmax=52,
            nconmax=15,
            cone="pyramidal",
            impratio=1,
            integrator="implicitfast",
        ),
        num_substeps=1,
        debug_mode=False,
    )
    feather_pgs = NewtonCfg(
        solver_cfg=FeatherPGSSolverCfg(
            angular_damping=0.2,
            update_mass_matrix_interval=1,
            enable_joint_limits=True,
            pgs_iterations=16,
            pgs_beta=0.02,
            pgs_cfm=1.0e-5,
            pgs_omega=0.8,
            dense_max_constraints=64,
            pgs_warmstart=False,
            pgs_mode="split",
            mf_max_constraints=512,
        ),
        collision_cfg=NewtonCollisionPipelineCfg(max_triangle_pairs=2_500_000),
        num_substeps=1,
        debug_mode=False,
        use_cuda_graph=False,
        default_shape_cfg=NewtonShapeCfg(margin=0.01),
    )
    physx = default
    newton_kamino = NewtonCfg(solver_cfg=KaminoSolverCfg(max_contacts_per_world=64), num_substeps=2)


@configclass
class CassieFlatEnvCfg(CassieRoughEnvCfg):
    sim: SimulationCfg = SimulationCfg(physics=PhysicsCfg())

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
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
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
