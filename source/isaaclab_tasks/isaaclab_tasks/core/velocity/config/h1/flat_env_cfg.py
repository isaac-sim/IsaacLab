# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_newton.physics import (
    FeatherPGSSolverCfg,
    KaminoSolverCfg,
    MJWarpSolverCfg,
    NewtonCfg,
)
from isaaclab_physx.physics import PhysxCfg

from isaaclab.sim import SimulationCfg
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.utils import PresetCfg

from .rough_env_cfg import H1RoughEnvCfg


@configclass
class PhysicsCfg(PresetCfg):
    default = PhysxCfg(gpu_max_rigid_patch_count=10 * 2**15)
    newton_mjwarp = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            use_mujoco_contacts=False,
            njmax=65,
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
            angular_damping=0.05,
            pgs_mode="matrix_free",
            update_mass_matrix_interval=1,
            enable_joint_limits=True,
            joint_limit_activation_gap=0.1,
            pgs_iterations=8,
            pgs_velocity_iterations=0,
            dense_max_constraints=64,
            mf_max_constraints=512,
            pgs_warmstart=False,
            pgs_omega=1.0,
            pgs_velocity_drive_mode="freeze",
            pgs_beta=0.05,
            pgs_cfm=1.0e-6,
            serial_kernel_block_dim=64,
        ),
        num_substeps=1,
        debug_mode=False,
        use_cuda_graph=True,
    )
    physx = default
    newton_kamino = NewtonCfg(solver_cfg=KaminoSolverCfg(max_contacts_per_world=64))


@configclass
class H1FlatEnvCfg(H1RoughEnvCfg):
    sim: SimulationCfg = SimulationCfg(physics=PhysicsCfg())

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None
        self.rewards.feet_air_time.weight = 1.0
        self.rewards.feet_air_time.params["threshold"] = 0.6


class H1FlatEnvCfg_PLAY(H1FlatEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.events.base_external_force_torque = None
        self.events.push_robot = None
