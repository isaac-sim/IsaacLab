# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
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

from isaaclab_assets.robots.humanoid import HUMANOID_CFG


@configclass
class HumanoidPhysicsCfg(PresetCfg):
    isaacsim_physx: PhysxCfg = PhysxCfg(bounce_threshold_velocity=0.2)
    ovphysx: OvPhysxCfg = OvPhysxCfg()
    physx: PhysxAutoCfg = PhysxAutoCfg(isaacsim_physx=isaacsim_physx, ovphysx=ovphysx)
    default = isaacsim_physx
    newton_mjwarp: NewtonCfg = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            njmax=80,
            nconmax=25,
            cone="pyramidal",
            update_data_interval=2,
            integrator="implicitfast",
            impratio=1,
        ),
        num_substeps=2,
        debug_mode=False,
    )


@configclass
class HumanoidEnvCfg(DirectRLEnvCfg):
    """Configuration for the direct-workflow Humanoid walking environment."""

    # env
    episode_length_s = 16.0
    decimation = 2
    action_scale = 1.0
    action_space = 21
    observation_space = 87
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation, physics=HumanoidPhysicsCfg())
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
    robot: ArticulationCfg = HUMANOID_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # effort scale per joint, keyed by joint name expression
    joint_gears: dict[str, float] = {
        ".*_waist.*": 67.5,
        ".*_upper_arm.*": 67.5,
        "pelvis": 67.5,
        ".*_lower_arm": 45.0,
        ".*_thigh:0": 45.0,
        ".*_thigh:1": 135.0,
        ".*_thigh:2": 45.0,
        ".*_shin": 90.0,
        ".*_foot.*": 22.5,
    }

    # sensors
    joint_wrench: JointWrenchSensorCfg = JointWrenchSensorCfg(prim_path="/World/envs/env_.*/Robot")
    feet_body_names: list[str] = ["left_foot", "right_foot"]

    # walk target, relative to the environment origin
    target_pos: tuple[float, float, float] = (1000.0, 0.0, 0.0)

    # reset
    initial_joint_pos_range: tuple[float, float] = (-0.2, 0.2)  # [rad]
    initial_joint_vel_range: tuple[float, float] = (-0.1, 0.1)  # [rad/s]

    heading_weight: float = 0.5
    up_weight: float = 0.1

    energy_cost_scale: float = 0.005
    actions_cost_scale: float = 0.01
    alive_reward_scale: float = 2.0
    joint_pos_limits_cost_scale: float = 0.25
    joint_pos_limits_threshold: float = 0.98

    death_cost: float = -1.0
    termination_height: float = 0.8

    # observation scales
    dof_vel_scale: float = 0.1
    angular_velocity_scale: float = 0.25
    contact_force_scale: float = 0.01
