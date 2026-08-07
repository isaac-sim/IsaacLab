# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from isaaclab_newton.physics import KaminoSolverCfg, MJWarpSolverCfg, NewtonCfg
from isaaclab_ovphysx.physics import OvPhysxCfg
from isaaclab_physx.physics import PhysxCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.physics import PhysxAutoCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import JointWrenchSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.utils import PresetCfg

from isaaclab_assets.robots.ant import ANT_CFG


@configclass
class AntPhysicsCfg(PresetCfg):
    isaacsim_physx: PhysxCfg = PhysxCfg(bounce_threshold_velocity=0.2)
    ovphysx: OvPhysxCfg = OvPhysxCfg()
    physx: PhysxAutoCfg = PhysxAutoCfg(isaacsim_physx=isaacsim_physx, ovphysx=ovphysx)
    default = isaacsim_physx
    newton_mjwarp: NewtonCfg = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            njmax=45,
            nconmax=25,
            cone="pyramidal",
            integrator="implicitfast",
            impratio=1,
        ),
        num_substeps=1,
        debug_mode=False,
    )
    newton_kamino: NewtonCfg = NewtonCfg(
        solver_cfg=KaminoSolverCfg(
            integrator="moreau",
            use_collision_detector=False,
            sparse_jacobian=True,
            constraints_alpha=0.1,
            padmm_max_iterations=100,
            padmm_primal_tolerance=1e-4,
            padmm_dual_tolerance=1e-4,
            padmm_compl_tolerance=1e-4,
            padmm_rho_0=0.05,
            padmm_eta=1e-5,
            padmm_use_acceleration=True,
            padmm_warmstart_mode="containers",
            padmm_contact_warmstart_method="geom_pair_net_force",
            padmm_use_graph_conditionals=False,
            collision_detector_pipeline="unified",
            collision_detector_max_contacts_per_pair=8,
        ),
        debug_mode=False,
        use_cuda_graph=True,
    )


@configclass
class AntEnvCfg(DirectRLEnvCfg):
    """Configuration for the direct-workflow Ant walking environment."""

    # env
    episode_length_s = 16.0
    decimation = 2
    action_scale = 0.5
    action_space = 8
    observation_space = 60
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation, physics=AntPhysicsCfg())
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="average",
            restitution_combine_mode="average",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=5.0, replicate_physics=True, clone_in_fabric=True
    )

    # robot
    robot: ArticulationCfg = ANT_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    # effort scale per joint, keyed by joint name expression
    joint_gears: dict[str, float] = {".*": 15.0}

    # sensors
    joint_wrench: JointWrenchSensorCfg = JointWrenchSensorCfg(prim_path="/World/envs/env_.*/Robot")
    feet_body_names: list[str] = ["front_left_foot", "front_right_foot", "left_back_foot", "right_back_foot"]

    # walk target, relative to the environment origin
    target_pos: tuple[float, float, float] = (1000.0, 0.0, 0.0)

    # reset
    initial_joint_pos_range: tuple[float, float] = (-0.2, 0.2)  # [rad]
    initial_joint_vel_range: tuple[float, float] = (-0.1, 0.1)  # [rad/s]

    heading_weight: float = 0.5
    up_weight: float = 0.1

    energy_cost_scale: float = 0.05
    actions_cost_scale: float = 0.005
    alive_reward_scale: float = 0.5
    joint_pos_limits_cost_scale: float = 0.1
    joint_pos_limits_threshold: float = 0.99

    death_cost: float = -2.0
    termination_height: float = 0.31

    # observation scales
    dof_vel_scale: float = 0.2
    angular_velocity_scale: float = 1.0
    contact_force_scale: float = 0.1
